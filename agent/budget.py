# -*- encoding: utf-8 -*-
"""预算守卫模块。

监控和限制 GPU 时间、磁盘空间、内存使用，确保实验不超出资源预算。
"""

import os
import psutil
import shutil
from typing import Optional
from datetime import datetime


class BudgetExceededError(Exception):
    """预算超出异常。"""
    pass


class ResourceError(Exception):
    """资源不足异常。"""
    pass


class BudgetGuard:
    """预算守卫。

    跟踪和限制实验资源使用。
    """

    def __init__(self,
                 max_gpu_hours: float = 38.0,
                 max_disk_gb: float = 100.0,
                 min_free_disk_gb: float = 10.0,
                 min_free_memory_gb: float = 4.0,
                 max_api_cost_usd: float = 50.0):
        """初始化预算守卫。

        Args:
            max_gpu_hours: 最大 GPU 小时数
            max_disk_gb: 最大磁盘使用量（GB）
            min_free_disk_gb: 最小剩余磁盘空间（GB）
            min_free_memory_gb: 最小剩余内存（GB）
            max_api_cost_usd: 最大 API 成本（美元）
        """
        self.max_gpu_hours = max_gpu_hours
        self.max_disk_gb = max_disk_gb
        self.min_free_disk_gb = min_free_disk_gb
        self.min_free_memory_gb = min_free_memory_gb
        self.max_api_cost_usd = max_api_cost_usd

        # 使用统计
        self.used_gpu_hours = 0.0
        self.used_disk_gb = 0.0
        self.used_api_cost_usd = 0.0

        # 记录
        self.action_log = []

    def check_gpu_budget(self, estimated_gpu_hours: float) -> bool:
        """检查 GPU 预算。

        Args:
            estimated_gpu_hours: 预计使用的 GPU 小时数

        Returns:
            是否有足够预算

        Raises:
            BudgetExceededError: 预算不足时抛出
        """
        total = self.used_gpu_hours + estimated_gpu_hours

        if total > self.max_gpu_hours:
            raise BudgetExceededError(
                f"GPU budget exceeded: {total:.1f} / {self.max_gpu_hours:.1f} hours. "
                f"Action requires {estimated_gpu_hours:.1f} hours."
            )

        return True

    def check_api_budget(self, estimated_cost_usd: float) -> bool:
        """检查 API 预算。

        Args:
            estimated_cost_usd: 预计 API 成本（美元）

        Returns:
            是否有足够预算

        Raises:
            BudgetExceededError: 预算不足时抛出
        """
        total = self.used_api_cost_usd + estimated_cost_usd

        if total > self.max_api_cost_usd:
            raise BudgetExceededError(
                f"API budget exceeded: ${total:.2f} / ${self.max_api_cost_usd:.2f}. "
                f"Action requires ${estimated_cost_usd:.2f}."
            )

        return True

    def check_disk_space(self, path: str = ".") -> bool:
        """检查磁盘空间。

        Args:
            path: 要检查的路径

        Returns:
            是否有足够空间

        Raises:
            ResourceError: 空间不足时抛出
        """
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)

        if free_gb < self.min_free_disk_gb:
            raise ResourceError(
                f"Insufficient disk space: {free_gb:.1f} GB free, "
                f"minimum required: {self.min_free_disk_gb:.1f} GB"
            )

        return True

    def check_memory(self) -> bool:
        """检查可用内存。

        Returns:
            是否有足够内存

        Raises:
            ResourceError: 内存不足时抛出
        """
        mem = psutil.virtual_memory()
        free_gb = mem.available / (1024 ** 3)

        if free_gb < self.min_free_memory_gb:
            raise ResourceError(
                f"Insufficient memory: {free_gb:.1f} GB available, "
                f"minimum required: {self.min_free_memory_gb:.1f} GB"
            )

        return True

    def check_all(self,
                  estimated_gpu_hours: float = 0.0,
                  estimated_cost_usd: float = 0.0,
                  check_disk: bool = True,
                  check_memory: bool = True) -> bool:
        """检查所有资源限制。

        Args:
            estimated_gpu_hours: 预计 GPU 小时数
            estimated_cost_usd: 预计 API 成本
            check_disk: 是否检查磁盘
            check_memory: 是否检查内存

        Returns:
            所有检查是否通过

        Raises:
            BudgetExceededError 或 ResourceError
        """
        if estimated_gpu_hours > 0:
            self.check_gpu_budget(estimated_gpu_hours)

        if estimated_cost_usd > 0:
            self.check_api_budget(estimated_cost_usd)

        if check_disk:
            self.check_disk_space()

        if check_memory:
            self.check_memory()

        return True

    def record_gpu_usage(self, gpu_hours: float, action_id: str = ""):
        """记录 GPU 使用。

        Args:
            gpu_hours: 实际使用的 GPU 小时数
            action_id: 动作标识符
        """
        self.used_gpu_hours += gpu_hours
        self.action_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "gpu",
            "value": gpu_hours,
            "action_id": action_id
        })

    def record_api_cost(self, cost_usd: float, action_id: str = ""):
        """记录 API 成本。

        Args:
            cost_usd: API 成本（美元）
            action_id: 动作标识符
        """
        self.used_api_cost_usd += cost_usd
        self.action_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "api",
            "value": cost_usd,
            "action_id": action_id
        })

    def get_remaining_budget(self) -> dict:
        """获取剩余预算。

        Returns:
            剩余预算字典
        """
        return {
            "gpu_hours": {
                "used": round(self.used_gpu_hours, 2),
                "max": self.max_gpu_hours,
                "remaining": round(self.max_gpu_hours - self.used_gpu_hours, 2),
                "percentage": round(self.used_gpu_hours / self.max_gpu_hours * 100, 1)
            },
            "api_cost_usd": {
                "used": round(self.used_api_cost_usd, 2),
                "max": self.max_api_cost_usd,
                "remaining": round(self.max_api_cost_usd - self.used_api_cost_usd, 2),
                "percentage": round(self.used_api_cost_usd / self.max_api_cost_usd * 100, 1)
            }
        }

    def print_summary(self):
        """打印预算使用摘要。"""
        remaining = self.get_remaining_budget()

        print("\n" + "=" * 60)
        print("Budget Summary")
        print("=" * 60)

        print(f"\nGPU Hours:")
        print(f"  Used:      {remaining['gpu_hours']['used']:.2f} / {remaining['gpu_hours']['max']:.2f}")
        print(f"  Remaining: {remaining['gpu_hours']['remaining']:.2f}")
        print(f"  Usage:     {remaining['gpu_hours']['percentage']:.1f}%")

        print(f"\nAPI Cost (USD):")
        print(f"  Used:      ${remaining['api_cost_usd']['used']:.2f} / ${remaining['api_cost_usd']['max']:.2f}")
        print(f"  Remaining: ${remaining['api_cost_usd']['remaining']:.2f}")
        print(f"  Usage:     {remaining['api_cost_usd']['percentage']:.1f}%")

        print("=" * 60 + "\n")


# Seed 白名单（全局常量）
ALLOWED_SEEDS = [0, 1, 42]


def validate_seed(seed: int) -> bool:
    """验证 seed 是否在白名单中。

    Args:
        seed: 随机种子

    Returns:
        是否有效

    Raises:
        ValueError: seed 不在白名单时抛出
    """
    if seed not in ALLOWED_SEEDS:
        raise ValueError(
            f"Seed {seed} not in whitelist. "
            f"Allowed seeds: {ALLOWED_SEEDS}"
        )
    return True
