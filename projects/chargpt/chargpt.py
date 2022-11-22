"""
Trains a character-level language model.
"""

import os
import sys

import time

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
# from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_default_config():
    C = CN()
    # device to train on
    C.device = 'auto'
    # dataloder parameters
    C.num_workers = 4
    # optimizer parameters
    C.max_iters = None
    C.batch_size = 64
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

# -----------------------------------------------------------------------------

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

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('data/tinyshakespeare.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    config.model.model_type = 'gpt2'
    model = GPT(config.model)

    # construct the trainer object
    # trainer = Trainer(config.trainer, model, train_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # # iteration callback
    # def batch_end_callback(trainer):

    #     if trainer.iter_num % 10 == 0:
    #         print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    #     if trainer.iter_num % 500 == 0:
    #         # evaluate both the train and test score
    #         model.eval()
    #         with torch.no_grad():
    #             # sample from the model...
    #             context = "O God, O God!"
    #             x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    #             y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
    #             completion = ''.join([train_dataset.itos[int(i)] for i in y])
    #             print(completion)
    #         # save the latest model
    #         print("saving model")
    #         ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    #         torch.save(model.state_dict(), ckpt_path)
    #         # revert model to training mode
    #         model.train()

    # trainer.set_callback('on_batch_end', batch_end_callback)


    # setup the optimizer
    optimizer = model.configure_optimizers(config.trainer)

    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
    )

    model.train()
    iter_num = 0
    iter_time = time.time()
    data_iter = iter(train_loader)
    while True:

        # fetch the next batch (x, y) and re-init iterator if needed
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = [t.to(device) for t in batch]
        x, y = batch

        # forward the model
        logits, loss = model(x, y)

        # backprop and update the parameters
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.grad_norm_clip)
        optimizer.step()

        # self.trigger_callbacks('on_batch_end')
        iter_num += 1
        tnow = time.time()
        iter_dt = tnow - iter_time
        iter_time = tnow

        # termination conditions
        if config.trainer.max_iters is not None and iter_num >= config.trainer.max_iters:
            break


if __name__ == '__main__':
    main()
