"""OpenCSLR 设备与分布式运行时管理。"""

import os

import numpy as np
import torch
import torch.distributed as dist


class DeviceManager:
    """管理设备、torchrun 进程组及递归的数据迁移。"""

    gpu_list = []
    output_device = torch.device("cpu")
    rank = 0
    local_rank = 0
    world_size = 1
    is_distributed = False

    @classmethod
    def init(cls, device=None):
        """根据 torchrun 环境初始化当前进程设备和进程组。"""
        cls.rank = int(os.environ.get("RANK", "0"))
        cls.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        cls.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        cls.is_distributed = cls.world_size > 1

        requested = "None" if device is None else str(device).strip()
        use_cpu = requested.lower() in {"none", "cpu", ""} or not torch.cuda.is_available()
        if use_cpu:
            if cls.is_distributed and not dist.is_initialized():
                dist.init_process_group(
                    backend=os.environ.get("DIST_BACKEND", "gloo"),
                    init_method="env://",
                    rank=cls.rank,
                    world_size=cls.world_size,
                )
            cls.gpu_list = []
            cls.output_device = torch.device("cpu")
            return

        if cls.is_distributed:
            if cls.local_rank >= torch.cuda.device_count():
                raise ValueError(f"LOCAL_RANK {cls.local_rank} exceeds available CUDA devices")
            torch.cuda.set_device(cls.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=os.environ.get("DIST_BACKEND", "nccl"),
                    init_method="env://",
                    rank=cls.rank,
                    world_size=cls.world_size,
                )
            cls.gpu_list = [cls.local_rank]
            cls.output_device = torch.device("cuda", cls.local_rank)
            return

        device_ids = [item.strip() for item in requested.split(",") if item.strip()]
        if any(not item.isdigit() for item in device_ids):
            raise ValueError(f"Invalid CUDA device specification: {device!r}")
        if any(int(item) >= torch.cuda.device_count() for item in device_ids):
            raise ValueError(f"CUDA device {device!r} is unavailable")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
        cls.gpu_list = list(range(len(device_ids)))
        cls.output_device = torch.device("cuda:0")
        torch.cuda.set_device(0)

    @classmethod
    def is_main_process(cls):
        return cls.rank == 0

    @classmethod
    def barrier(cls):
        if cls.is_distributed and dist.is_initialized():
            dist.barrier()

    @classmethod
    def cleanup(cls):
        if cls.is_distributed and dist.is_initialized():
            dist.destroy_process_group()

    @classmethod
    def to(cls, data):
        """将 Tensor、NumPy 数组或嵌套序列迁移到当前设备。"""
        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data)
        if isinstance(data, torch.Tensor):
            if data.dtype == torch.float64:
                data = data.float()
            elif data.dtype == torch.uint8:
                data = data.long()
            return data.to(cls.output_device, non_blocking=True)
        if isinstance(data, (list, tuple)):
            converted = [cls.to(item) for item in data]
            return converted if isinstance(data, list) else tuple(converted)
        raise TypeError(f"Unsupported data type: {type(data).__name__}")
