# -*- encoding: utf-8 -*-
"""随机种子管理工具。

提供统一的随机种子设置接口，确保所有随机数生成器（Python、NumPy、PyTorch、CUDA）
使用相同的种子，实现确定性复现。
"""

import random
import numpy as np
import torch


def set_seed(seed, rank=0, deterministic=True):
    """设置全局随机种子，确保实验的确定性复现。

    统一设置 Python 内置 random、NumPy、PyTorch CPU 和 CUDA 的随机种子。
    为多进程训练（DDP）预留 rank 参数，不同进程使用不同但确定的种子。

    Args:
        seed (int): 基础随机种子，默认 0
        rank (int): 进程编号，用于 DDP 训练，默认 0
        deterministic (bool): 是否启用 cuDNN 确定性模式。
            True: 完全确定性，但可能降低性能
            False: 允许 cuDNN 自动调优，性能更好但结果可能有微小差异

    Example:
        >>> # 单卡训练
        >>> set_seed(0)

        >>> # 多卡训练，每个进程使用不同种子
        >>> set_seed(0, rank=local_rank)

    Note:
        - 单 seed 结果用于确定性复现和系统比较，不用于估计运行波动或统计显著性
        - deterministic=True 可能降低 10-20% 训练速度
        - 即使固定种子，不同 CUDA/cuDNN 版本或硬件可能产生微小差异（通常 <0.3% WER）
    """
    # 计算当前进程的实际种子
    effective_seed = seed + rank

    # Python 内置随机数生成器
    random.seed(effective_seed)

    # NumPy 随机数生成器
    np.random.seed(effective_seed)

    # PyTorch CPU 随机数生成器
    torch.manual_seed(effective_seed)

    # PyTorch CUDA 随机数生成器（所有 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(effective_seed)
        torch.cuda.manual_seed_all(effective_seed)

    # cuDNN 确定性设置
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 允许 cuDNN 自动调优，更快但可能不完全确定
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_rng_state():
    """获取当前所有随机数生成器的状态。

    用于保存检查点时记录随机状态，以便断点续训时精确恢复。

    Returns:
        dict: 包含所有随机数生成器状态的字典
            - 'python': Python random 模块的状态
            - 'numpy': NumPy 随机数生成器的状态
            - 'torch': PyTorch CPU 随机数生成器的状态
            - 'cuda': PyTorch CUDA 随机数生成器的状态（所有 GPU）

    Example:
        >>> rng_state = get_rng_state()
        >>> torch.save({'model': model.state_dict(), 'rng': rng_state}, 'checkpoint.pt')
    """
    rng_state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_state['cuda'] = torch.cuda.get_rng_state_all()
    else:
        rng_state['cuda'] = None

    return rng_state


def set_rng_state(rng_state):
    """恢复随机数生成器的状态。

    从保存的状态字典中恢复所有随机数生成器，用于断点续训。

    Args:
        rng_state (dict): 由 get_rng_state() 返回的状态字典

    Example:
        >>> checkpoint = torch.load('checkpoint.pt')
        >>> set_rng_state(checkpoint['rng'])
        >>> # 现在随机状态已恢复，训练将产生与中断前完全相同的结果

    Note:
        确保在加载模型权重后、开始训练前调用此函数
    """
    # Python random
    random.setstate(rng_state['python'])

    # NumPy
    np.random.set_state(rng_state['numpy'])

    # PyTorch CPU
    # 如果 rng_state 从 GPU 加载，需要先移到 CPU
    torch_state = rng_state['torch']
    if torch_state.is_cuda:
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)

    # PyTorch CUDA
    if rng_state['cuda'] is not None and torch.cuda.is_available():
        cuda_states = rng_state['cuda']
        # 确保所有状态都在 CPU 上
        cuda_states = [s.cpu() if s.is_cuda else s for s in cuda_states]
        torch.cuda.set_rng_state_all(cuda_states)


def seed_worker(worker_id):
    """DataLoader worker 初始化函数，确保每个 worker 使用独立但确定的种子。

    传递给 torch.utils.data.DataLoader 的 worker_init_fn 参数，
    确保多个 worker 之间的随机操作（如数据增强）是确定的且互不干扰。

    Args:
        worker_id (int): DataLoader worker 编号

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker
        ... )

    Note:
        这确保了即使使用多个 DataLoader worker，数据增强仍然是确定的
    """
    # 从 PyTorch 的初始种子派生 worker 种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
