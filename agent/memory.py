# -*- encoding: utf-8 -*-
"""实验记忆模块。

记录所有实验的配置、结果、成本，支持去重和查询。
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class ExperimentMemory:
    """实验记忆系统。

    存储实验历史，支持配置去重和结果查询。
    """

    def __init__(self, memory_file: str = "agent/memory.json"):
        """初始化记忆系统。

        Args:
            memory_file: 记忆文件路径
        """
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载已有记忆
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Any] = {
            "total_experiments": 0,
            "total_gpu_hours": 0.0,
            "total_api_cost": 0.0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

        if self.memory_file.exists():
            self.load()

    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置的哈希值。

        忽略随机种子、工作目录等不影响结果的字段。

        Args:
            config: 配置字典

        Returns:
            哈希值（16 字符）
        """
        # 移除不影响结果的字段
        config_copy = config.copy()
        ignore_keys = [
            "work_dir", "random_seed", "print_log",
            "log_interval", "save_interval", "wandb"
        ]

        for key in ignore_keys:
            config_copy.pop(key, None)

        # 序列化并计算哈希
        config_str = json.dumps(config_copy, sort_keys=True)
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()[:16]

    def has_experiment(self, config: Dict[str, Any]) -> bool:
        """检查实验是否已运行。

        Args:
            config: 配置字典

        Returns:
            是否已存在
        """
        config_hash = self.compute_config_hash(config)
        return config_hash in self.experiments

    def get_experiment(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取已有实验结果。

        Args:
            config: 配置字典

        Returns:
            实验结果（如果存在）
        """
        config_hash = self.compute_config_hash(config)
        return self.experiments.get(config_hash)

    def add_experiment(self,
                      config: Dict[str, Any],
                      results: Dict[str, Any],
                      efficiency: Optional[Dict[str, Any]] = None,
                      api_cost: float = 0.0,
                      gpu_hours: float = 0.0):
        """添加实验记录。

        Args:
            config: 配置字典
            results: 结果（WER、状态等）
            efficiency: 效率指标（参数量、显存等）
            api_cost: API 成本
            gpu_hours: GPU 小时数
        """
        config_hash = self.compute_config_hash(config)

        self.experiments[config_hash] = {
            "config_hash": config_hash,
            "config": config,
            "results": results,
            "efficiency": efficiency or {},
            "api_cost": api_cost,
            "gpu_hours": gpu_hours,
            "timestamp": datetime.now().isoformat()
        }

        # 更新元数据
        self.metadata["total_experiments"] = len(self.experiments)
        self.metadata["total_gpu_hours"] += gpu_hours
        self.metadata["total_api_cost"] += api_cost
        self.metadata["last_updated"] = datetime.now().isoformat()

        # 自动保存
        self.save()

    def query_experiments(self,
                         dataset: Optional[str] = None,
                         model: Optional[str] = None,
                         protocol: Optional[str] = None,
                         min_wer: Optional[float] = None,
                         max_wer: Optional[float] = None) -> List[Dict[str, Any]]:
        """查询实验。

        Args:
            dataset: 数据集过滤
            model: 模型过滤
            protocol: 协议过滤
            min_wer: 最小 WER 过滤
            max_wer: 最大 WER 过滤

        Returns:
            匹配的实验列表
        """
        results = []

        for exp in self.experiments.values():
            config = exp["config"]
            exp_results = exp["results"]

            # 应用过滤器
            if dataset and config.get("dataset") != dataset:
                continue

            if model and config.get("model") != model:
                continue

            if protocol and config.get("protocol") != protocol:
                continue

            wer = exp_results.get("wer") or exp_results.get("dev_wer")
            if wer is not None:
                if min_wer is not None and wer < min_wer:
                    continue
                if max_wer is not None and wer > max_wer:
                    continue

            results.append(exp)

        # 按 WER 排序
        results.sort(key=lambda x: x["results"].get("wer", float("inf")))

        return results

    def get_best_experiment(self,
                           dataset: Optional[str] = None,
                           model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取最佳实验。

        Args:
            dataset: 数据集过滤
            model: 模型过滤

        Returns:
            最佳实验（WER 最低）
        """
        experiments = self.query_experiments(dataset=dataset, model=model)
        return experiments[0] if experiments else None

    def get_cost_per_wer_point(self,
                              baseline_wer: float,
                              current_wer: float,
                              total_cost: float) -> float:
        """计算每个 WER 点改进的成本。

        Args:
            baseline_wer: 基线 WER
            current_wer: 当前 WER
            total_cost: 总成本（GPU 小时或美元）

        Returns:
            每 WER 点成本
        """
        improvement = baseline_wer - current_wer
        if improvement <= 0:
            return float("inf")

        return total_cost / improvement

    def save(self):
        """保存记忆到文件。"""
        data = {
            "experiments": self.experiments,
            "metadata": self.metadata
        }

        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self):
        """从文件加载记忆。"""
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.experiments = data.get("experiments", {})
        self.metadata = data.get("metadata", {})

    def export_summary(self, output_path: str):
        """导出摘要报告。

        Args:
            output_path: 输出文件路径
        """
        summary = {
            "metadata": self.metadata,
            "top_experiments": self.query_experiments()[:10],
            "by_dataset": {},
            "by_model": {}
        }

        # 按数据集汇总
        for exp in self.experiments.values():
            dataset = exp["config"].get("dataset")
            if dataset:
                if dataset not in summary["by_dataset"]:
                    summary["by_dataset"][dataset] = []
                summary["by_dataset"][dataset].append(exp)

        # 按模型汇总
        for exp in self.experiments.values():
            model = exp["config"].get("model")
            if model:
                if model not in summary["by_model"]:
                    summary["by_model"][model] = []
                summary["by_model"][model].append(exp)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """打印摘要统计。"""
        print("\n" + "=" * 60)
        print("Experiment Memory Summary")
        print("=" * 60)
        print(f"Total experiments: {self.metadata['total_experiments']}")
        print(f"Total GPU hours:   {self.metadata['total_gpu_hours']:.1f}")
        print(f"Total API cost:    ${self.metadata['total_api_cost']:.2f}")

        if self.experiments:
            best = self.get_best_experiment()
            if best:
                print(f"\nBest experiment:")
                print(f"  WER: {best['results'].get('wer', 'N/A')}")
                print(f"  Dataset: {best['config'].get('dataset')}")
                print(f"  Model: {best['config'].get('model')}")

        print("=" * 60 + "\n")
