# -*- encoding: utf-8 -*-
"""样本统计与错误记录工具。

用于记录实验中的样本处理情况，包括成功样本、跳过样本、失败样本，
以及详细的错误信息。确保实验透明度和可调试性。
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SampleError:
    """单个样本的错误记录。"""
    sample_id: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class SampleStatistics:
    """样本统计跟踪器。

    记录实验中所有样本的处理状态，包括：
    - 成功处理的样本
    - 跳过的样本（数据缺失等）
    - 失败的样本（运行时错误）

    自动计算跳过率并判断实验是否有效（skip_rate ≤ 5%）。
    """

    def __init__(self, total_samples: int, experiment_name: str = ""):
        """初始化样本统计器。

        Args:
            total_samples: 数据集中的总样本数
            experiment_name: 实验名称，用于记录
        """
        self.total_samples = total_samples
        self.experiment_name = experiment_name

        self.successful_samples: List[str] = []
        self.skipped_samples: List[str] = []
        self.failed_samples: List[str] = []

        self.skip_reasons: Dict[str, List[str]] = {}  # reason -> [sample_ids]
        self.errors: List[SampleError] = []

    def record_success(self, sample_id: str):
        """记录成功处理的样本。

        Args:
            sample_id: 样本标识符
        """
        self.successful_samples.append(sample_id)

    def record_skip(self, sample_id: str, reason: str = "unknown"):
        """记录跳过的样本。

        Args:
            sample_id: 样本标识符
            reason: 跳过原因（如 "file_not_found", "corrupted_video"）
        """
        self.skipped_samples.append(sample_id)
        if reason not in self.skip_reasons:
            self.skip_reasons[reason] = []
        self.skip_reasons[reason].append(sample_id)

    def record_failure(self, sample_id: str, error: Exception,
                      include_traceback: bool = True):
        """记录失败的样本。

        Args:
            sample_id: 样本标识符
            error: 捕获的异常对象
            include_traceback: 是否包含完整的堆栈跟踪
        """
        self.failed_samples.append(sample_id)

        error_record = SampleError(
            sample_id=sample_id,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc() if include_traceback else None,
            timestamp=datetime.now().isoformat()
        )
        self.errors.append(error_record)

    @property
    def num_successful(self) -> int:
        """成功样本数。"""
        return len(self.successful_samples)

    @property
    def num_skipped(self) -> int:
        """跳过样本数。"""
        return len(self.skipped_samples)

    @property
    def num_failed(self) -> int:
        """失败样本数。"""
        return len(self.failed_samples)

    @property
    def skip_rate(self) -> float:
        """跳过率（跳过样本数 / 总样本数）。"""
        if self.total_samples == 0:
            return 0.0
        return self.num_skipped / self.total_samples

    @property
    def failure_rate(self) -> float:
        """失败率（失败样本数 / 总样本数）。"""
        if self.total_samples == 0:
            return 0.0
        return self.num_failed / self.total_samples

    @property
    def is_valid(self) -> bool:
        """实验是否有效（skip_rate ≤ 5%）。"""
        return self.skip_rate <= 0.05

    @property
    def status(self) -> str:
        """实验状态：'valid' 或 'invalid'。"""
        return "valid" if self.is_valid else "invalid"

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要。

        Returns:
            包含所有统计信息的字典
        """
        return {
            "experiment": self.experiment_name,
            "total_samples": self.total_samples,
            "successful": self.num_successful,
            "skipped": self.num_skipped,
            "failed": self.num_failed,
            "skip_rate": round(self.skip_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "status": self.status,
            "skip_reasons": {
                reason: len(samples)
                for reason, samples in self.skip_reasons.items()
            }
        }

    def get_detailed_report(self,
                           include_sample_ids: bool = True,
                           include_tracebacks: bool = False) -> Dict[str, Any]:
        """获取详细报告。

        Args:
            include_sample_ids: 是否包含所有样本 ID 列表
            include_tracebacks: 是否包含完整的错误堆栈

        Returns:
            详细报告字典
        """
        report = self.get_summary()

        if include_sample_ids:
            report["successful_samples"] = self.successful_samples
            report["skipped_samples"] = self.skipped_samples
            report["failed_samples"] = self.failed_samples
            report["skip_reasons_detail"] = self.skip_reasons

        if self.errors:
            report["errors"] = [
                {
                    "sample_id": err.sample_id,
                    "error_type": err.error_type,
                    "error_message": err.error_message,
                    "timestamp": err.timestamp,
                    **({"traceback": err.traceback} if include_tracebacks else {})
                }
                for err in self.errors
            ]

        return report

    def save(self, output_path: str, detailed: bool = True):
        """保存统计报告到 JSON 文件。

        Args:
            output_path: 输出文件路径
            detailed: 是否保存详细报告（包含样本 ID 和错误详情）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if detailed:
            report = self.get_detailed_report(
                include_sample_ids=True,
                include_tracebacks=True
            )
        else:
            report = self.get_summary()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """打印统计摘要到控制台。"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"Sample Statistics: {self.experiment_name}")
        print("=" * 60)
        print(f"Total samples:      {summary['total_samples']}")
        print(f"Successful:         {summary['successful']} "
              f"({summary['successful']/summary['total_samples']*100:.1f}%)")
        print(f"Skipped:            {summary['skipped']} "
              f"({summary['skip_rate']*100:.2f}%)")
        print(f"Failed:             {summary['failed']} "
              f"({summary['failure_rate']*100:.2f}%)")
        print(f"Status:             {summary['status'].upper()}")

        if self.skip_reasons:
            print("\nSkip reasons:")
            for reason, count in summary['skip_reasons'].items():
                print(f"  - {reason}: {count}")

        if not self.is_valid:
            print(f"\n⚠️  WARNING: Skip rate ({summary['skip_rate']*100:.2f}%) "
                  f"exceeds 5% threshold!")
            print("    This experiment is marked INVALID.")

        print("=" * 60 + "\n")

    @staticmethod
    def load(input_path: str) -> 'SampleStatistics':
        """从 JSON 文件加载统计数据。

        Args:
            input_path: 输入文件路径

        Returns:
            SampleStatistics 对象
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stats = SampleStatistics(
            total_samples=data['total_samples'],
            experiment_name=data.get('experiment', '')
        )

        stats.successful_samples = data.get('successful_samples', [])
        stats.skipped_samples = data.get('skipped_samples', [])
        stats.failed_samples = data.get('failed_samples', [])
        stats.skip_reasons = data.get('skip_reasons_detail', {})

        # 重建错误记录
        if 'errors' in data:
            stats.errors = [
                SampleError(**err_data)
                for err_data in data['errors']
            ]

        return stats


def create_sample_manifest(stats_list: List[SampleStatistics],
                          output_path: str,
                          require_unanimous: bool = True):
    """从多个实验的统计数据创建有效样本清单。

    确保所有模型在相同的有效样本集上进行评估。

    Args:
        stats_list: 多个实验的 SampleStatistics 对象列表
        output_path: 输出清单文件路径
        require_unanimous: 是否要求所有实验都成功处理的样本才算有效
                          True: 取交集（最严格）
                          False: 只要有一个实验成功就算有效（最宽松）

    Returns:
        有效样本 ID 的集合
    """
    if not stats_list:
        raise ValueError("stats_list cannot be empty")

    if require_unanimous:
        # 取交集：所有实验都成功的样本
        valid_samples = set(stats_list[0].successful_samples)
        for stats in stats_list[1:]:
            valid_samples &= set(stats.successful_samples)
    else:
        # 取并集：至少一个实验成功的样本
        valid_samples = set()
        for stats in stats_list:
            valid_samples |= set(stats.successful_samples)

    # 保存清单
    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_experiments": len(stats_list),
        "experiments": [s.experiment_name for s in stats_list],
        "require_unanimous": require_unanimous,
        "num_valid_samples": len(valid_samples),
        "valid_samples": sorted(list(valid_samples))
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Sample manifest created: {len(valid_samples)} valid samples")
    print(f"Saved to: {output_path}")

    return valid_samples


def filter_dataset_by_manifest(dataset, manifest_path: str):
    """根据样本清单过滤数据集。

    Args:
        dataset: 数据集对象（需要有 sample_ids 属性或方法）
        manifest_path: 样本清单文件路径

    Returns:
        过滤后的数据集索引列表
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    valid_samples = set(manifest['valid_samples'])

    # 假设 dataset 有 get_sample_id(idx) 方法或 sample_ids 列表
    if hasattr(dataset, 'sample_ids'):
        sample_ids = dataset.sample_ids
    elif hasattr(dataset, 'get_sample_id'):
        sample_ids = [dataset.get_sample_id(i) for i in range(len(dataset))]
    else:
        raise ValueError("Dataset must have 'sample_ids' or 'get_sample_id' method")

    # 找出有效样本的索引
    valid_indices = [
        i for i, sid in enumerate(sample_ids)
        if sid in valid_samples
    ]

    print(f"Filtered dataset: {len(valid_indices)}/{len(sample_ids)} samples")

    return valid_indices
