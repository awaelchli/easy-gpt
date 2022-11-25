"""
Trains a character-level language model.
"""
from dataclasses import dataclass
import time
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT, Block
import functools
from lightning_lite import seed_everything
from lightning_lite.lite import LightningLite
from lightning_lite.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload, BackwardPrefetch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

@dataclass
class GPTConfig:
    model_type: str
    vocab_size: int
    block_size: int
    embd_pdrop: float =  0.1
    resid_pdrop: float =  0.1
    attn_pdrop: float =  0.1
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd:  Optional[int] = None

    def __post_init__(self):
        type_given = self.model_type is not None
        params_given = all((self.n_layer is not None, self.n_head is not None, self.n_embd is not None))
        assert type_given ^ params_given
        if type_given:
            # translate from model_type to detailed configuration
            values = {
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[self.model_type]
            self.n_layer=values["n_layer"]
            self.n_head=values["n_head"]
            self.n_embd=values["n_embd"]

@dataclass
class TrainerConfig:
    block_size: int
    num_workers: int
    batch_size: int
    learning_rate: float
    betas: Tuple[int]
    weight_decay: float
    grad_norm_clip: float
    seed: int = 1
    max_iters: int = -1




model_config = GPTConfig(
    model_type = 'gpt2-xl',
    vocab_size = None,
    block_size =  128,
    embd_pdrop = 0.1,
    resid_pdrop = 0.1,
    attn_pdrop = 0.1,
)


trainer_config = TrainerConfig(
    num_workers = 4,
    max_iters = 100,
    block_size = 128,
    batch_size = 2,
    learning_rate = 3e-4,
    betas = (0.9, 0.95),
    weight_decay = 0.1, # only applied on matmul weights
    grad_norm_clip = 1.0,
)


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    def __init__(self, data, block_size):
        self.block_size = block_size
        chars = sorted(list(set(data)))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
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
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def main():
    seed_everything(trainer_config.seed)

    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    check_fn = lambda submodule: isinstance(submodule, Block)
    wrapper = functools.partial(checkpoint_wrapper, offload_to_cpu=False, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    # TODO: precision 16 and cpu offload hangs
    # TODO: error messaging for cpu-offload + wrap policy
    lite = LightningLite(accelerator="cuda", devices=4, precision=16, strategy=FSDPStrategy(auto_wrap_policy=auto_wrap_policy, backward_prefetch=BackwardPrefetch.BACKWARD_PRE))
    lite.launch()

    # construct the training dataset
    text = open('data/tinyshakespeare.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, block_size=model_config.block_size)

    # construct the model
    model_config.vocab_size = train_dataset.get_vocab_size()

    lite.print(model_config)
    lite.print(trainer_config)

    # setup the model and optimizer
    with lite.sharded_model():
        model = GPT(model_config)
    model = lite.setup_module(model)

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)

    # TODO: support multiple param groups for FSDP
    # optimizer = model.configure_optimizers(config.trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config.learning_rate, betas=trainer_config.betas)
    optimizer = lite.setup_optimizers(optimizer)


    # setup the dataloader
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
    iter_num = 0
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

        # forward the model
        logits, loss = model(x, y)

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        lite.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_norm_clip)
        optimizer.step()

        if iter_num % 10 == 0:
            lite.print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss.item():.5f}")

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if trainer_config.max_iters != -1 and iter_num >= trainer_config.max_iters:
            break

    # For optimal memory throughput, make sure the summary shows 0 cudaMalloc retries and otherwise try lowering the batch size.
    print(torch.cuda.memory_summary())


if __name__ == '__main__':
    main()
