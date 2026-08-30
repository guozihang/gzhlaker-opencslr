# -*- encoding: utf-8 -*-
"""Dataloader manager module for OpenCSLR.

Creates and manages PyTorch DataLoader instances for training, validation,
and test data splits using the dataset instances and collate function.
"""

import torch
from torch.utils.data import DataLoader
from .argument_manager import ArgumentManager
from .dataset_manager import DatasetManager
from .collect_manager import CollectManager

class DataloaderManager:
    """
    The static class that handle the dataloader object
    """
    DATALOADER = {}

    @classmethod
    def init(cls):
        """Initialize DataLoader instances for all dataset splits.

        Creates a DataLoader for each dataset mode (train, train_eval, dev, test)
        with appropriate batch sizes, shuffle settings, and collate function.
        """
        arg = ArgumentManager.get( )
        dataset_list = DatasetManager.get_dataset_list()
        for idx, (mode, train_flag) in enumerate(dataset_list):
            cls.DATALOADER [ mode ] = torch.utils.data.DataLoader(
                DatasetManager.get(mode),
                batch_size=arg.batch_size if mode == "train" else arg.test_batch_size,
                shuffle=train_flag,
                drop_last=train_flag,
                num_workers=arg.num_worker,  # if train_flag else 0
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
        cls.DATALOADER[mode]
