"""Optional length-aware batch sampling for non-DDP training."""

import random

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    """Group nearby sequence lengths to reduce padding waste."""

    def __init__(self, lengths, batch_size, bucket_size=0, shuffle=True, drop_last=True):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.bucket_size = max(int(bucket_size), self.batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __iter__(self):
        order = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(order)
        bucket_width = self.bucket_size * self.batch_size
        buckets = [order[i:i + bucket_width] for i in range(0, len(order), bucket_width)]
        batches = []
        for bucket in buckets:
            bucket.sort(key=self.lengths.__getitem__, reverse=True)
            batches.extend(
                bucket[i:i + self.batch_size]
                for i in range(0, len(bucket), self.batch_size)
                if len(bucket[i:i + self.batch_size]) == self.batch_size or not self.drop_last
            )
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        bucket_width = self.bucket_size * self.batch_size
        total = 0
        for start in range(0, len(self.lengths), bucket_width):
            bucket_length = min(bucket_width, len(self.lengths) - start)
            total += bucket_length // self.batch_size if self.drop_last else (
                bucket_length + self.batch_size - 1
            ) // self.batch_size
        return total
