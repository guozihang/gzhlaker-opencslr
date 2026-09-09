# -*- encoding: utf-8 -*-
"""实验结果汇总与比较工具。

提供标准化的结果记录、加载和比较功能，确保所有实验结果格式统一，
便于生成论文表格和进行公平比较。
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd


@dataclass
class ExperimentResult:
    """单个实验的完整结果记录。"""
    # 实验标识
    experiment_name: str
    protocol: str  # 'unified' or 'official'
    seed: int
    dataset: str
    split: str  # 'train', 'dev', 'test'

    # 模型信息
    model: str
    model_config: Optional[Dict[str, Any]] = None

    # 解码设置
    decoder: str = "greedy"  # 'greedy' or 'beam-{size}'
    decoder_config: Optional[Dict[str, Any]] = None

    # 性能指标
    wer: Optional[float] = None
    substitutions: Optional[int] = None
    insertions: Optional[int] = None
    deletions: Optional[int] = None
    reference_length: Optional[int] = None

    # 样本统计
    total_samples: Optional[int] = None
    successful_samples: Optional[int] = None
    skipped_samples: Optional[int] = None
    failed_samples: Optional[int] = None
    skip_rate: Optional[float] = None
    status: Optional[str] = None  # 'valid' or 'invalid'

    # 效率指标
    params_m: Optional[float] = None  # 参数量（百万）
    peak_memory_gb: Optional[float] = None  # GPU 显存峰值
    train_throughput: Optional[float] = None  # 训练吞吐（samples/sec）
    train_time_hours: Optional[float] = None  # 训练总时长（GPU·小时）
    inference_ms_per_video: Optional[float] = None  # 单视频推理时间（毫秒）

    # 元数据
    timestamp: Optional[str] = None
    git_commit: Optional[str] = None
    work_dir: Optional[str] = None
    config_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return asdict(self)

    def to_table_row(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """转换为表格行格式。

        Args:
            columns: 需要包含的列名列表，None 表示包含所有非 None 字段

        Returns:
            适合制表的字典
        """
        if columns is None:
            # 默认表格列
            columns = [
                'model', 'dataset', 'split', 'protocol',
                'wer', 'decoder', 'status',
                'params_m', 'peak_memory_gb', 'inference_ms_per_video'
            ]

        row = {}
        for col in columns:
            value = getattr(self, col, None)
            if value is not None:
                row[col] = value

        return row


class ResultAggregator:
    """实验结果聚合器。

    收集、存储、比较多个实验的结果，支持生成统一格式的表格。
    """

    def __init__(self, output_dir: str = "./results"):
        """初始化结果聚合器。

        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[ExperimentResult] = []

    def add_result(self, result: ExperimentResult):
        """添加一个实验结果。

        Args:
            result: ExperimentResult 对象
        """
        if result.timestamp is None:
            result.timestamp = datetime.now().isoformat()
        self.results.append(result)

    def add_from_dict(self, result_dict: Dict[str, Any]):
        """从字典添加结果。

        Args:
            result_dict: 包含结果字段的字典
        """
        result = ExperimentResult(**result_dict)
        self.add_result(result)

    def add_from_json(self, json_path: str):
        """从 JSON 文件加载结果。

        Args:
            json_path: JSON 文件路径
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # 批量加载
            for item in data:
                self.add_from_dict(item)
        else:
            # 单个结果
            self.add_from_dict(data)

    def filter_results(self,
                      protocol: Optional[str] = None,
                      dataset: Optional[str] = None,
                      split: Optional[str] = None,
                      status: str = "valid") -> List[ExperimentResult]:
        """过滤结果。

        Args:
            protocol: 协议筛选（'unified' 或 'official'）
            dataset: 数据集筛选
            split: 数据集分割筛选
            status: 有效性筛选（'valid' 或 'invalid'），None 表示不过滤

        Returns:
            过滤后的结果列表
        """
        filtered = self.results

        if protocol:
            filtered = [r for r in filtered if r.protocol == protocol]

        if dataset:
            filtered = [r for r in filtered if r.dataset == dataset]

        if split:
            filtered = [r for r in filtered if r.split == split]

        if status:
            filtered = [r for r in filtered if r.status == status]

        return filtered

    def generate_table(self,
                      protocol: str = "unified",
                      dataset: Optional[str] = None,
                      split: str = "test",
                      sort_by: str = "wer",
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """生成结果对比表格。

        Args:
            protocol: 协议筛选
            dataset: 数据集筛选（None 表示所有数据集）
            split: 数据集分割
            sort_by: 排序字段
            columns: 包含的列

        Returns:
            pandas DataFrame
        """
        filtered = self.filter_results(
            protocol=protocol,
            dataset=dataset,
            split=split,
            status="valid"
        )

        if not filtered:
            print(f"No valid results found for {protocol}/{dataset}/{split}")
            return pd.DataFrame()

        rows = [r.to_table_row(columns) for r in filtered]
        df = pd.DataFrame(rows)

        if sort_by in df.columns:
            df = df.sort_values(by=sort_by)

        return df

    def save_table(self,
                  output_path: str,
                  format: str = "csv",
                  **kwargs):
        """保存表格到文件。

        Args:
            output_path: 输出文件路径
            format: 输出格式（'csv', 'json', 'latex', 'markdown'）
            **kwargs: 传递给 generate_table 的参数
        """
        df = self.generate_table(**kwargs)

        if df.empty:
            print(f"No data to save for {kwargs}")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient='records', indent=2)
        elif format == "latex":
            latex_str = df.to_latex(index=False, float_format="%.2f")
            with open(output_path, 'w') as f:
                f.write(latex_str)
        elif format == "markdown":
            md_str = df.to_markdown(index=False, floatfmt=".2f")
            with open(output_path, 'w') as f:
                f.write(md_str)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Table saved to {output_path}")

    def save_all_results(self, output_path: str):
        """保存所有原始结果到 JSON。

        Args:
            output_path: 输出 JSON 文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in self.results]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(data)} results to {output_path}")

    def print_summary(self):
        """打印结果汇总统计。"""
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Total experiments: {len(self.results)}")

        # 按协议统计
        protocols = {}
        for r in self.results:
            protocols[r.protocol] = protocols.get(r.protocol, 0) + 1
        print("\nBy protocol:")
        for protocol, count in protocols.items():
            print(f"  - {protocol}: {count}")

        # 按数据集统计
        datasets = {}
        for r in self.results:
            datasets[r.dataset] = datasets.get(r.dataset, 0) + 1
        print("\nBy dataset:")
        for dataset, count in datasets.items():
            print(f"  - {dataset}: {count}")

        # 按状态统计
        statuses = {}
        for r in self.results:
            statuses[r.status or 'unknown'] = statuses.get(r.status or 'unknown', 0) + 1
        print("\nBy status:")
        for status, count in statuses.items():
            print(f"  - {status}: {count}")

        print("=" * 60 + "\n")

    def compare_models(self,
                      model_names: List[str],
                      dataset: str,
                      split: str = "test",
                      protocol: str = "unified") -> pd.DataFrame:
        """比较指定模型的性能。

        Args:
            model_names: 要比较的模型名称列表
            dataset: 数据集名称
            split: 数据集分割
            protocol: 实验协议

        Returns:
            比较表格（DataFrame）
        """
        filtered = self.filter_results(
            protocol=protocol,
            dataset=dataset,
            split=split,
            status="valid"
        )

        filtered = [r for r in filtered if r.model in model_names]

        if not filtered:
            print(f"No results found for comparison")
            return pd.DataFrame()

        rows = [r.to_table_row() for r in filtered]
        df = pd.DataFrame(rows)

        return df.sort_values(by='wer')


def save_experiment_result(result: ExperimentResult, output_path: str):
    """保存单个实验结果。

    Args:
        result: ExperimentResult 对象
        output_path: 输出 JSON 文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def load_experiment_result(input_path: str) -> ExperimentResult:
    """加载单个实验结果。

    Args:
        input_path: 输入 JSON 文件路径

    Returns:
        ExperimentResult 对象
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return ExperimentResult(**data)


def generate_paper_table(results_json: str,
                        protocol: str = "unified",
                        datasets: Optional[List[str]] = None,
                        output_format: str = "latex") -> str:
    """生成论文格式的结果表格。

    Args:
        results_json: 结果 JSON 文件路径
        protocol: 实验协议
        datasets: 数据集列表（None 表示所有）
        output_format: 输出格式（'latex' 或 'markdown'）

    Returns:
        格式化的表格字符串
    """
    aggregator = ResultAggregator()
    aggregator.add_from_json(results_json)

    if datasets is None:
        # 获取所有数据集
        datasets = list(set(r.dataset for r in aggregator.results))

    tables = []
    for dataset in datasets:
        df = aggregator.generate_table(
            protocol=protocol,
            dataset=dataset,
            split="test",
            columns=['model', 'wer', 'params_m', 'peak_memory_gb',
                    'inference_ms_per_video']
        )

        if not df.empty:
            if output_format == "latex":
                table_str = df.to_latex(index=False, float_format="%.2f")
            else:  # markdown
                table_str = df.to_markdown(index=False, floatfmt=".2f")

            tables.append(f"\n### {dataset}\n{table_str}")

    return "\n".join(tables)
