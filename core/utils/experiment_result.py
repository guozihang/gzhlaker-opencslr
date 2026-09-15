# -*- encoding: utf-8 -*-
"""实验结果记录。

把一次评估的 WER 与样本统计写成一个统一的 JSON 文件,供论文表格与
多实验比较使用。字段是纯数据,不依赖 pandas 等分析库。
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentResult:
    """单个实验在某个 split 上的结果记录。

    只填充有值的字段;无法获取的指标保持 None,写入 JSON 时为 null。
    """

    # 实验标识
    experiment_name: str
    seed: int
    dataset: str
    split: str  # 'dev' / 'test'

    # 模型与解码
    model: str
    decoder: str = "greedy"  # 'greedy' 或 'beam-{size}'

    # 性能指标
    wer: float = None

    # 样本统计(由 SampleStatistics 提供)
    total_samples: int = None
    successful_samples: int = None
    skipped_samples: int = None
    failed_samples: int = None
    skip_rate: float = None
    status: str = None  # 'valid' / 'invalid'

    # 元数据
    timestamp: str = None
    work_dir: str = None
    config_path: str = None

    def to_dict(self):
        """转换为可直接 json.dump 的字典。"""
        return asdict(self)


def save_experiment_result(result, output_path):
    """把实验结果写入 JSON,自动创建父目录。

    Args:
        result: ExperimentResult 实例。
        output_path: 输出文件路径。

    Returns:
        pathlib.Path: 实际写入的路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    return output_path
