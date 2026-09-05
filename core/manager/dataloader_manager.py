# -*- encoding: utf-8 -*-
"""Dataloader manager module for OpenCSLR.

Creates and manages PyTorch DataLoader instances for training, validation,
and test data splits using the dataset instances and collate function.
"""

import torch
import cv2
import os
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .argument_manager import ArgumentManager
from .dataset_manager import DatasetManager
from .collect_manager import CollectManager
from .device_manager import DeviceManager
from .cuda_prefetcher import CUDAPrefetcher
from .length_bucket_sampler import LengthBucketBatchSampler

class DataloaderManager:
    """
    The static class that handle the dataloader object
    """
    DATALOADER = {}
    SAMPLER = {}

    @staticmethod
    def _worker_init(worker_id, threads):
        """Prevent each IO worker from spawning its own CPU thread pool."""
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    @classmethod
    def init(cls):
        """Initialize DataLoader instances for all dataset splits.

        Creates a DataLoader for each dataset mode (train, train_eval, dev, test)
        with appropriate batch sizes, shuffle settings, and collate function.
        Uses DistributedSampler when running in distributed mode.
        """
        arg = ArgumentManager.get()
        dataset_list = DatasetManager.get_dataset_list()
        cls.DATALOADER.clear()
        cls.SAMPLER.clear()
        for idx, (mode, train_flag) in enumerate(dataset_list):
            dataset = DatasetManager.get(mode)
            sampler = None
            shuffle = train_flag
            # 仅训练集做分布式切分；评估集保持完整数据，由 rank 0 统一评估。
            batch_size = arg.batch_size if mode == "train" else arg.test_batch_size
            batch_sampler = None
            if (mode == "train" and not DeviceManager.is_distributed
                    and getattr(arg, "length_bucket_size", 0) > 0
                    and hasattr(dataset, "sample_lengths")):
                batch_sampler = LengthBucketBatchSampler(
                    dataset.sample_lengths(), batch_size, arg.length_bucket_size,
                    shuffle=True, drop_last=True,
                )
                shuffle = False
            elif DeviceManager.is_distributed and mode == "train":
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=DeviceManager.world_size,
                    rank=DeviceManager.rank,
                    shuffle=train_flag,
                    drop_last=train_flag,
                )
                shuffle = False
                cls.SAMPLER[mode] = sampler
            num_workers = arg.num_worker if mode == "train" else arg.eval_num_worker
            loader_options = dict(
                dataset=dataset,
                collate_fn=CollectManager.collate,
                pin_memory=bool(arg.pin_memory),
                worker_init_fn=partial(cls._worker_init, threads=max(1, int(arg.worker_threads))),
                num_workers=num_workers,
            )
            if batch_sampler is not None:
                loader_options["batch_sampler"] = batch_sampler
            else:
                loader_options.update(
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    drop_last=train_flag if sampler is None else False,
                )
            if num_workers > 0:
                loader_options["prefetch_factor"] = max(1, int(arg.prefetch_factor))
                loader_options["persistent_workers"] = bool(arg.persistent_workers and mode == "train")
            cls.DATALOADER[mode] = DataLoader(
                **loader_options,
            )

    @classmethod
    def get_iterator(cls, mode="train"):
        """Return a device-prefetching iterator when CUDA is enabled."""
        arg = ArgumentManager.get()
        gpu_augment = None
        dataset = DatasetManager.get(mode)
        if getattr(arg, "gpu_prefetch", False) and getattr(dataset, "gpu_augment", False):
            from libs.gpu_video_augmentation import GPUVideoAugment
            gpu_augment = GPUVideoAugment(
                image_scale=dataset.image_scale,
                training=dataset.transform_mode == "train",
            )
        return CUDAPrefetcher(cls.get(mode), DeviceManager.output_device, gpu_augment)

    @classmethod
    def shutdown(cls):
        """Release DataLoader worker processes at program shutdown."""
        for loader in cls.DATALOADER.values():
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()
        cls.DATALOADER.clear()

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
