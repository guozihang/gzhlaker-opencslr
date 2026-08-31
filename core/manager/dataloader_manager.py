# -*- encoding: utf-8 -*-
"""Dataloader manager module for OpenCSLR.

Creates and manages PyTorch DataLoader instances for training, validation,
and test data splits using the dataset instances and collate function.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .argument_manager import ArgumentManager
from .dataset_manager import DatasetManager
from .collect_manager import CollectManager
from .device_manager import DeviceManager

class DataloaderManager:
    """
    The static class that handle the dataloader object
    """
    DATALOADER = {}
    SAMPLER = {}

    @classmethod
    def init(cls):
        """Initialize DataLoader instances for all dataset splits.

        Creates a DataLoader for each dataset mode (train, train_eval, dev, test)
        with appropriate batch sizes, shuffle settings, and collate function.
        Uses DistributedSampler when running in distributed mode.
        """
        arg = ArgumentManager.get()
        dataset_list = DatasetManager.get_dataset_list()
        for idx, (mode, train_flag) in enumerate(dataset_list):
            dataset = DatasetManager.get(mode)
            sampler = None
            shuffle = train_flag
            # 仅训练集做分布式切分；评估集保持完整数据，由 rank 0 统一评估。
            if DeviceManager.is_distributed and mode == "train":
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=DeviceManager.world_size,
                    rank=DeviceManager.rank,
                    shuffle=train_flag,
                    drop_last=train_flag,
                )
                shuffle = False
                cls.SAMPLER[mode] = sampler
            cls.DATALOADER[mode] = DataLoader(
                dataset,
                batch_size=arg.batch_size if mode == "train" else arg.test_batch_size,
                shuffle=shuffle,
                sampler=sampler,
                drop_last=train_flag if sampler is None else False,
                num_workers=arg.num_worker,
                collate_fn=CollectManager.collate,
                pin_memory=True,
            )

    @classmethod
    def get(cls, mode="train"):
        """Get the DataLoader for the specified mode.

        Args:
            mode: Dataset mode, one of "train", "train_eval", "dev", "test"

        Returns:
            DataLoader: The requested DataLoader instance
        """
        return cls.DATALOADER[mode]

    @classmethod
    def set_epoch(cls, mode, epoch):
        """Set the epoch for the sampler to ensure proper shuffling in distributed mode.

        Args:
            mode: Dataset mode
            epoch: Current epoch number
        """
        if mode in cls.SAMPLER:
            cls.SAMPLER[mode].set_epoch(epoch)
