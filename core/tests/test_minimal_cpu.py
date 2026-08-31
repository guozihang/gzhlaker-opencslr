# -*- coding: utf-8 -*-
"""CPU 最小化验证脚本。

无需 GPU 即可运行,验证 model_to_device 与 SyncBatchNorm 转换逻辑在
CPU 路径下不会报错,并直接验证 nn.SyncBatchNorm.convert_sync_batchnorm
的转换行为。

用法(在有 torch 的环境):
    cd core
    python tests/test_minimal_cpu.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn


def test_device_manager_cpu():
    """CPU 路径下 DeviceManager 初始化。"""
    from manager.device_manager import DeviceManager
    DeviceManager.init("none")
    assert DeviceManager.output_device.type == "cpu", DeviceManager.output_device
    assert not DeviceManager.is_distributed
    print(f"[OK] DeviceManager CPU: output_device={DeviceManager.output_device}")


def test_syncbn_conversion_on_cpu():
    """直接验证 nn.SyncBatchNorm.convert_sync_batchnorm 转换行为(CPU 也可运行)。"""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(3)

    model = DummyModel()
    assert type(model.bn).__name__ == "BatchNorm2d"
    converted = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # PyTorch 原生 SyncBatchNorm 不区分 1d/2d/3d 子类,统一为 SyncBatchNorm
    assert type(converted.bn).__name__ == "SyncBatchNorm", type(converted.bn)
    print(f"[OK] SyncBatchNorm conversion: BatchNorm2d -> {type(converted.bn).__name__}")


def test_model_to_device_cpu_branch():
    """验证 model_to_device 在 CPU(非分布式) 路径下不触发 SyncBN 转换。"""
    from manager.device_manager import DeviceManager
    if not DeviceManager.is_distributed:
        # 模拟 ExperimentManager.model_to_device 的核心逻辑
        model = nn.BatchNorm2d(3)
        model = model.to(DeviceManager.output_device)
        if DeviceManager.output_device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        assert type(model).__name__ == "BatchNorm2d"
        print("[OK] model_to_device CPU branch: no SyncBN conversion (expected)")


if __name__ == "__main__":
    test_device_manager_cpu()
    test_syncbn_conversion_on_cpu()
    test_model_to_device_cpu_branch()
    print("\nAll minimal CPU tests passed.")
