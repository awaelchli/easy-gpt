"""
Trains a character-level language model.
"""
import functools
import time


import torch
from lightning_lite import seed_everything
from lightning_lite.lite import LightningLite
from lightning_lite.strategies.fsdp import FSDPStrategy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT, Block
from mingpt.config import GPTConfig, TrainerConfig


model_config = GPTConfig(
    model_type="gpt2-xl",
    vocab_size=None,
    block_size=128,
    embd_pdrop=0.1,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
)


trainer_config = TrainerConfig(
    num_workers=4,
    max_iters=100,
    block_size=128,
    batch_size=64,
    learning_rate=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,  # only applied on matmul weights
    grad_norm_clip=1.0,
)


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def main():
    seed_everything(trainer_config.seed)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    check_fn = lambda submodule: isinstance(submodule, Block)
    wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    # TODO: precision 16 and cpu offload hangs
    lite = LightningLite(
        accelerator="cuda",
        devices=4,
        precision=16,
        strategy=FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        ),
    )
    lite.launch()

    # construct the training dataset
    text = open(
        "data/tinyshakespeare.txt", "r"
    ).read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, block_size=model_config.block_size)

    # construct the model
    model_config.vocab_size = train_dataset.get_vocab_size()

    lite.print(model_config)
    lite.print(trainer_config)

    # setup the model and optimizer
    with lite.sharded_model():
        model = GPT(model_config)
    model = lite.setup_module(model)

    lite.print(f"Number of parameters: {model.num_parameters / 1e6:.1f} M")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn
    )

    # TODO: support multiple param groups for FSDP
    # optimizer = model.configure_optimizers(config.trainer)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, betas=trainer_config.betas
    )
    optimizer = lite.setup_optimizers(optimizer)

    train_loader = DataLoader(
        train_dataset,
        # TODO: fix this in Lite
        # sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=True,
        pin_memory=True,
        batch_size=trainer_config.batch_size,
        num_workers=trainer_config.num_workers,
    )
    train_loader = lite.setup_dataloaders(train_loader)

    model.train()
    iteration = 0
    iter_dt = 0
    iter_time = time.time()
    data_iter = iter(train_loader)

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y = batch

        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        lite.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), trainer_config.grad_norm_clip
        )
        optimizer.step()

        if iteration % 10 == 0:
            lite.print(
                f"iteration time {iter_dt * 1000:.2f}ms; iteration {iteration}; train loss {loss.item():.5f}"
            )

        iteration += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        if trainer_config.max_iters != -1 and iteration >= trainer_config.max_iters:
            break

    # For optimal memory throughput, make sure the summary shows 0 cudaMalloc retries and otherwise try lowering the batch size.
    lite.print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
