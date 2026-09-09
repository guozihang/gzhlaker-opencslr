# -*- encoding: utf-8 -*-
"""OpenCSLR 环境包装器。

将 core/main.py 封装为可程序化调用的环境，供 Agent Harness 使用。
不修改 core/ 任何代码，纯粹通过 CLI 和配置文件交互。
"""

import os
import sys
import json
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class OpenCSLREnv:
    """OpenCSLR 环境包装器。

    提供 Agent 与 OpenCSLR 交互的标准接口：
    - 评估已训练模型（改变解码器设置）
    - 生成新配置文件
    - 提交训练任务到队列
    - 解析结果
    """

    def __init__(self, core_dir: str = "./core", work_root: str = "./work_dir"):
        """初始化环境。

        Args:
            core_dir: core/ 目录路径
            work_root: 工作目录根路径
        """
        self.core_dir = Path(core_dir).resolve()
        self.work_root = Path(work_root).resolve()

        if not self.core_dir.exists():
            raise ValueError(f"Core directory not found: {self.core_dir}")

        self.work_root.mkdir(parents=True, exist_ok=True)

        # 缓存：避免重复读取配置
        self._config_cache: Dict[str, Dict] = {}

    def evaluate_decoder(self,
                        checkpoint_path: str,
                        dataset: str,
                        split: str,
                        decoder_config: Dict[str, Any],
                        work_dir: Optional[str] = None) -> Dict[str, Any]:
        """在已训练的 checkpoint 上评估不同的解码器配置。

        Args:
            checkpoint_path: 模型 checkpoint 路径
            dataset: 数据集名称
            split: 数据集分割（dev 或 test）
            decoder_config: 解码器配置（decode_mode, beam_size, lm_weight 等）
            work_dir: 输出目录（可选，自动生成）

        Returns:
            评估结果字典，包含 WER、时间、状态等
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 推断原始配置
        original_config_path = checkpoint_path.parent / "config.yaml"
        if not original_config_path.exists():
            # 尝试其他可能的位置
            original_config_path = self.core_dir / "configs" / f"unified_{dataset}.yaml"

        if not original_config_path.exists():
            raise FileNotFoundError(f"Cannot infer config for checkpoint: {checkpoint_path}")

        # 创建临时配置（覆盖解码器设置）
        with open(original_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 更新解码器配置
        if 'decoder_args' not in config:
            config['decoder_args'] = {}
        config['decoder_args'].update(decoder_config)

        # 设置为测试模式
        config['phase'] = 'test'
        config['dataset'] = dataset

        # 创建临时工作目录
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"agent_eval_{dataset}_{split}_", dir=self.work_root)
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        # 保存临时配置
        temp_config = Path(work_dir) / "temp_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)

        # 构建命令
        cmd = [
            sys.executable,
            str(self.core_dir / "main.py"),
            "--config", str(temp_config),
            "--phase", "test",
            "--load-weights", str(checkpoint_path),
            "--work-dir", str(work_dir),
            "--dataset", dataset,
        ]

        # 执行评估
        print(f"Running evaluation: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.core_dir
        )

        # 解析结果
        return self._parse_evaluation_result(work_dir, result)

    def _parse_evaluation_result(self, work_dir: Path, subprocess_result) -> Dict[str, Any]:
        """解析评估结果。

        Args:
            work_dir: 工作目录
            subprocess_result: subprocess 返回结果

        Returns:
            结构化的结果字典
        """
        result = {
            "status": "success" if subprocess_result.returncode == 0 else "failed",
            "returncode": subprocess_result.returncode,
            "stdout": subprocess_result.stdout,
            "stderr": subprocess_result.stderr,
        }

        # 尝试读取结果文件
        result_json = work_dir / "experiment_result.json"
        if result_json.exists():
            with open(result_json, 'r') as f:
                experiment_result = json.load(f)
                result.update({
                    "wer": experiment_result.get("wer"),
                    "total_samples": experiment_result.get("total_samples"),
                    "skip_rate": experiment_result.get("skip_rate"),
                })

        # 读取样本统计
        stats_json = work_dir / "sample_statistics.json"
        if stats_json.exists():
            with open(stats_json, 'r') as f:
                stats = json.load(f)
                result["sample_statistics"] = stats

        return result

    def generate_config(self,
                       base_config: str,
                       overrides: Dict[str, Any],
                       output_path: str) -> Path:
        """生成新的配置文件。

        Args:
            base_config: 基础配置文件路径
            overrides: 需要覆盖的配置项
            output_path: 输出配置路径

        Returns:
            生成的配置文件路径
        """
        # 读取基础配置
        base_config_path = Path(base_config)
        if not base_config_path.exists():
            base_config_path = self.core_dir / "configs" / base_config

        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 深度合并覆盖项
        config = self._deep_merge(config, overrides)

        # 写入新配置
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return output_path

    def _deep_merge(self, base: Dict, overrides: Dict) -> Dict:
        """深度合并字典。"""
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def submit_training(self,
                       config_path: str,
                       use_watchdog: bool = False,
                       gpus: Optional[List[int]] = None) -> Dict[str, Any]:
        """提交训练任务（通过 run.sh 或 watchdog）。

        Args:
            config_path: 配置文件路径
            use_watchdog: 是否使用 watchdog 自动重启
            gpus: GPU 设备列表

        Returns:
            提交信息
        """
        config_path = Path(config_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # 这里不实际启动训练，而是返回需要执行的命令
        # 实际执行由外部调度系统负责

        gpu_str = ",".join(map(str, gpus)) if gpus else "0"

        if use_watchdog:
            cmd = [
                "bash",
                "script/train_watchdog.sh",
                str(config_path),
                gpu_str
            ]
        else:
            cmd = [
                "bash",
                "script/run.sh",
                str(config_path),
                gpu_str
            ]

        return {
            "status": "queued",
            "command": " ".join(cmd),
            "config": str(config_path),
            "gpus": gpus
        }

    def parse_training_log(self, work_dir: str) -> Dict[str, Any]:
        """解析训练日志，提取 WER 等指标。

        Args:
            work_dir: 工作目录

        Returns:
            训练统计信息
        """
        work_dir = Path(work_dir)

        result = {
            "status": "unknown",
            "dev_wer_history": [],
            "test_wer": None,
            "best_epoch": None
        }

        # 尝试读取实验结果
        result_json = work_dir / "experiment_result.json"
        if result_json.exists():
            with open(result_json, 'r') as f:
                data = json.load(f)
                result["test_wer"] = data.get("wer")
                result["status"] = data.get("status", "completed")

        # 读取训练日志（如果存在）
        log_file = work_dir / "training.log"
        if log_file.exists():
            # 简单解析：提取每个 epoch 的 dev WER
            with open(log_file, 'r') as f:
                for line in f:
                    if "Dev WER" in line:
                        # 提取 WER 值（需要根据实际日志格式调整）
                        pass

        return result

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """获取 checkpoint 信息。

        Args:
            checkpoint_path: checkpoint 路径

        Returns:
            checkpoint 元信息
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 获取文件大小
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        # 尝试读取对应的配置
        config_path = checkpoint_path.parent / "config.yaml"
        config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        return {
            "path": str(checkpoint_path),
            "size_mb": round(size_mb, 2),
            "exists": True,
            "config": config
        }

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置的合法性。

        Args:
            config: 配置字典

        Returns:
            验证结果
        """
        try:
            # 尝试导入验证器
            sys.path.insert(0, str(self.core_dir))
            from utils.config_validator import ConfigValidator

            validator = ConfigValidator(config, verbose=False)
            is_valid, errors, warnings = validator.validate_all()

            return {
                "is_valid": is_valid,
                "errors": errors,
                "warnings": warnings
            }
        except ImportError:
            return {
                "is_valid": True,
                "errors": [],
                "warnings": ["Config validator not available"]
            }
