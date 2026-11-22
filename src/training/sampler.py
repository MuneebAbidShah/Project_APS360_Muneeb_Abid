"""Batch words of similar width together.

A batch is padded to its widest image, so mixing "a" with "understanding" wastes
most of the compute on padding. We shuffle, cut the data into big chunks, sort
each chunk by width, then slice it into batches and shuffle the batches. Every
epoch is still random, but there's much less padding (often 2x faster).
"""
import random

from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    def __init__(self, widths, batch_size, shuffle=True, chunk_batches=50, seed=0, drop_last=False):
        self.widths, self.batch_size, self.shuffle = list(widths), batch_size, shuffle
        self.chunk = batch_size * chunk_batches
        self.seed, self.epoch, self.drop_last = seed, 0, drop_last

    def set_epoch(self, epoch: int):
        self.epoch = epoch   # different shuffle each epoch, same across runs

    def _batches(self):
        idx = list(range(len(self.widths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        batches = []
        for start in range(0, len(idx), self.chunk):
            chunk = sorted(idx[start:start + self.chunk], key=lambda i: self.widths[i])
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b:b + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
