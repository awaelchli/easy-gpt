"""
Trains a character-level language model.
"""
import sys

import time

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.utils import setup_logging, CfgNode as CN
import functools
from lightning_lite import seed_everything
from lightning_lite.lite import LightningLite
from lightning_lite.strategies.fsdp import FSDPStrategy


def get_default_config():
    C = CN()
    # device to train on
    # dataloder parameters
    C.num_workers = 4
    # optimizer parameters
    C.max_iters = None
    C.batch_size = 16
    C.learning_rate = 3e-4
    C.betas = (0.9, 0.95)
    C.weight_decay = 0.1 # only applied on matmul weights
    C.grad_norm_clip = 1.0
    return C

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C


class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

def main():
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)
    from torch.distributed.fsdp import CPUOffload

    lite = LightningLite(accelerator="cuda", devices=4, precision=16, strategy=FSDPStrategy(cpu_offload=CPUOffload(offload_params=True)))
    lite.launch()

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    setup_logging(config)
    seed_everything(config.system.seed)

    # construct the training dataset
    text = open('data/tinyshakespeare.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    config.model.model_type = 'gpt2-medium'

    print(config)

    # setup the model and optimizer
    with lite.sharded_model():
        model = GPT(config.model)
    model = lite.setup_module(model)
    print("model", [p.device for p in model.parameters()])
    # TODO: support multiple param groups for FSDP
    # optimizer = model.configure_optimizers(config.trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.learning_rate, betas=config.trainer.betas)
    optimizer = lite.setup_optimizers(optimizer)

    print("optimizers", [p.device for p in model.parameters()])

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        # sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
    )

    train_loader = lite.setup_dataloaders(train_loader)

    model.train()
    iter_num = 0
    iter_dt = 0
    iter_time = time.time()
    data_iter = iter(train_loader)

    while True:

        # fetch the next batch (x, y) and re-init iterator if needed
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)
        optimizer.step()

        if iter_num % 10 == 0:
            print(f"iter_dt {iter_dt * 1000:.2f}ms; iter {iter_num}: train loss {loss.item():.5f}")

        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if config.trainer.max_iters is not None and iter_num >= config.trainer.max_iters:
            break


if __name__ == '__main__':
    main()
