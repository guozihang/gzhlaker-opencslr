# -*- encoding: utf-8 -*-
"""样本统计与错误记录工具。

记录一次评估中每个样本的处理结果(成功 / 跳过 / 失败)。跳过率超过阈值时
实验被判为 invalid,用于发现评估集里被静默丢掉的数据——否则 WER 会因为
"少算了一批难样本"而虚高。
"""

import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

SKIP_RATE_THRESHOLD = 0.05


@dataclass
class SampleError:
    """单个样本的处理失败记录。

    保存 traceback 是因为 seq_eval 只把异常摘要写进日志,完整的调用栈
    只有这里留了一份,排查坏样本时需要。
    """

    sample_id: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    timestamp: Optional[str] = None


class SampleStatistics:
    """样本统计跟踪器。

    记录成功、跳过、失败的样本 ID,自动计算跳过率并判断实验是否有效。
    """

    def __init__(self, total_samples: int, experiment_name: str = ""):
        """
        Args:
            total_samples: 数据集中的总样本数。
            experiment_name: 实验名称,写入保存的报告。
        """
        self.total_samples = total_samples
        self.experiment_name = experiment_name

        self.successful_samples: List[str] = []
        self.skipped_samples: List[str] = []
        self.failed_samples: List[str] = []

        self.skip_reasons: Dict[str, List[str]] = {}  # reason -> [sample_ids]
        self.errors: List[SampleError] = []

    def record_success(self, sample_id: str):
        """记录一个成功处理的样本。"""
        self.successful_samples.append(sample_id)

    def record_skip(self, sample_id: str, reason: str = "unknown"):
        """记录一个被跳过的样本。

        Args:
            sample_id: 样本标识符。
            reason: 跳过原因(如 "frames_exceeded")。
        """
        self.skipped_samples.append(sample_id)
        self.skip_reasons.setdefault(reason, []).append(sample_id)

    def record_failure(self, sample_id: str, error: Exception):
        """记录一个处理失败的样本。

        Args:
            sample_id: 样本标识符。
            error: 捕获到的异常对象。
        """
        self.failed_samples.append(sample_id)
        self.errors.append(SampleError(
            sample_id=sample_id,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            timestamp=datetime.now().isoformat(),
        ))

    @property
    def num_successful(self) -> int:
        return len(self.successful_samples)

    @property
    def num_skipped(self) -> int:
        return len(self.skipped_samples)

    @property
    def num_failed(self) -> int:
        return len(self.failed_samples)

    @property
    def skip_rate(self) -> float:
        """跳过率(跳过样本数 / 总样本数)。"""
        if self.total_samples == 0:
            return 0.0
        return self.num_skipped / self.total_samples

    @property
    def is_valid(self) -> bool:
        """跳过率未超过阈值时实验有效。"""
        return self.skip_rate <= SKIP_RATE_THRESHOLD

    @property
    def status(self) -> str:
        return "valid" if self.is_valid else "invalid"

    def get_summary(self) -> Dict[str, Any]:
        """获取不含明细的统计摘要。"""
        return {
            "experiment": self.experiment_name,
            "total_samples": self.total_samples,
            "successful": self.num_successful,
            "skipped": self.num_skipped,
            "failed": self.num_failed,
            "skip_rate": round(self.skip_rate, 4),
            "status": self.status,
            "skip_reasons": {
                reason: len(samples) for reason, samples in self.skip_reasons.items()
            },
        }

    def save(self, output_path):
        """把摘要与样本明细写入 JSON,自动创建父目录。

        Args:
            output_path: 输出文件路径。

        Returns:
            pathlib.Path: 实际写入的路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_summary()
        report["successful_samples"] = self.successful_samples
        report["skipped_samples"] = self.skipped_samples
        report["failed_samples"] = self.failed_samples
        report["skip_reasons_detail"] = self.skip_reasons
        report["errors"] = [asdict(err) for err in self.errors]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return output_path

    @staticmethod
    def load(input_path) -> "SampleStatistics":
        """从 ``save`` 产出的 JSON 恢复统计对象。"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stats = SampleStatistics(
            total_samples=data["total_samples"],
            experiment_name=data.get("experiment", ""),
        )
        stats.successful_samples = data.get("successful_samples", [])
        stats.skipped_samples = data.get("skipped_samples", [])
        stats.failed_samples = data.get("failed_samples", [])
        stats.skip_reasons = data.get("skip_reasons_detail", {})
        stats.errors = [SampleError(**err) for err in data.get("errors", [])]
        return stats

    def print_summary(self):
        """打印统计摘要到控制台。"""
        summary = self.get_summary()
        total = self.total_samples or 1

        print("\n" + "=" * 60)
        print(f"Sample Statistics: {self.experiment_name}")
        print("=" * 60)
        print(f"Total samples:      {summary['total_samples']}")
        print(f"Successful:         {summary['successful']} "
              f"({summary['successful'] / total * 100:.1f}%)")
        print(f"Skipped:            {summary['skipped']} ({summary['skip_rate'] * 100:.2f}%)")
        print(f"Failed:             {summary['failed']}")
        print(f"Status:             {summary['status'].upper()}")

        if self.skip_reasons:
            print("\nSkip reasons:")
            for reason, count in summary["skip_reasons"].items():
                print(f"  - {reason}: {count}")

        if not self.is_valid:
            print(f"\n⚠️  WARNING: Skip rate ({summary['skip_rate'] * 100:.2f}%) "
                  f"exceeds {SKIP_RATE_THRESHOLD * 100:.0f}% threshold!")
            print("    This experiment is marked INVALID.")

        print("=" * 60 + "\n")
