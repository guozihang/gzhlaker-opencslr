#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""协议合规性检查脚本。

检查配置文件、代码和实验结果是否符合 unified/official 协议规范。
用于验证实验是否满足 WWW 2027 投稿要求。
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 添加 core 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.config_validator import ConfigValidator


class ProtocolChecker:
    """协议合规性检查器。"""

    # Unified 协议要求
    UNIFIED_REQUIREMENTS = {
        'random_fix': True,
        'protocol': 'unified',
        'feeder_args': {
            'resize_shape': [256, 256],
            'crop_shape': [224, 224],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        },
        'decoder_args': {
            'decode_mode': 'greedy',  # 或 beam，但必须明确指定
        }
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues: List[str] = []
        self.warnings: List[str] = []

    def check_config(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """检查配置文件的协议合规性。

        Args:
            config_path: 配置文件路径

        Returns:
            (is_compliant, issues, warnings)
        """
        self.issues = []
        self.warnings = []

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        protocol = config.get('protocol', 'unified')

        if protocol == 'unified':
            self._check_unified_protocol(config)
        elif protocol == 'official':
            self._check_official_protocol(config)
        else:
            self.issues.append(f"Invalid protocol: {protocol}")

        is_compliant = len(self.issues) == 0

        if self.verbose:
            self._print_results(config_path)

        return is_compliant, self.issues, self.warnings

    def _check_unified_protocol(self, config: Dict[str, Any]):
        """检查 unified 协议的特定要求。"""
        # 1. 检查 random_fix
        if not config.get('random_fix', False):
            self.issues.append(
                "Unified protocol requires 'random_fix: true'"
            )

        # 2. 检查 random_seed 是否明确设置
        if 'random_seed' not in config:
            self.warnings.append(
                "random_seed not explicitly set (default 0 will be used)"
            )

        # 3. 检查预处理设置
        feeder_args = config.get('feeder_args', {})
        expected_preprocessing = self.UNIFIED_REQUIREMENTS['feeder_args']

        for key, expected_value in expected_preprocessing.items():
            actual_value = feeder_args.get(key)
            if actual_value != expected_value:
                self.issues.append(
                    f"Unified protocol requires feeder_args.{key} = {expected_value}, "
                    f"got {actual_value}"
                )

        # 4. 检查 decoder 是否明确指定
        decoder_args = config.get('decoder_args', {})
        if 'decode_mode' not in decoder_args:
            self.warnings.append(
                "decode_mode not specified in decoder_args"
            )

        # 5. 检查 cuDNN 确定性设置（通过 random_fix 控制）
        if config.get('random_fix') and 'cudnn_deterministic' in config:
            if not config['cudnn_deterministic']:
                self.warnings.append(
                    "random_fix is true but cudnn_deterministic is false. "
                    "This may cause slight non-determinism."
                )

    def _check_official_protocol(self, config: Dict[str, Any]):
        """检查 official 协议的要求。"""
        # Official 协议允许自定义设置，但仍需要记录
        if 'official_settings' not in config:
            self.warnings.append(
                "Official protocol should document custom settings in 'official_settings'"
            )

        # 仍然建议固定 seed
        if not config.get('random_fix', False):
            self.warnings.append(
                "Even for official protocol, recommend setting 'random_fix: true' "
                "for reproducibility"
            )

    def check_result(self, result_path: str) -> Tuple[bool, List[str], List[str]]:
        """检查结果文件的协议合规性。

        Args:
            result_path: 结果 JSON 文件路径

        Returns:
            (is_compliant, issues, warnings)
        """
        self.issues = []
        self.warnings = []

        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        # 1. 检查必需字段
        required_fields = [
            'experiment_name', 'protocol', 'seed', 'dataset', 'split',
            'model', 'decoder', 'wer'
        ]

        for field in required_fields:
            if field not in result:
                self.issues.append(f"Missing required field: {field}")

        # 2. 检查样本统计
        if 'total_samples' in result and 'skip_rate' in result:
            skip_rate = result['skip_rate']
            if skip_rate > 0.05:
                status = result.get('status', 'unknown')
                if status != 'invalid':
                    self.issues.append(
                        f"skip_rate ({skip_rate:.2%}) > 5% but status is not 'invalid'"
                    )
        else:
            self.warnings.append(
                "Sample statistics not found in result"
            )

        # 3. 检查协议标注
        protocol = result.get('protocol')
        if protocol not in ['unified', 'official']:
            self.issues.append(
                f"Invalid or missing protocol in result: {protocol}"
            )

        is_compliant = len(self.issues) == 0

        if self.verbose:
            self._print_results(result_path)

        return is_compliant, self.issues, self.warnings

    def _print_results(self, file_path: str):
        """打印检查结果。"""
        print("\n" + "=" * 70)
        print(f"Protocol Compliance Check: {Path(file_path).name}")
        print("=" * 70)

        if not self.issues and not self.warnings:
            print("✅ Fully compliant!")
        else:
            if self.issues:
                print(f"\n❌ {len(self.issues)} Compliance Issue(s):")
                for i, issue in enumerate(self.issues, 1):
                    print(f"  {i}. {issue}")

            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} Warning(s):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"  {i}. {warning}")

        print("=" * 70 + "\n")


def check_directory(directory: str, check_configs: bool = True,
                   check_results: bool = True) -> Dict[str, Any]:
    """批量检查目录中的所有配置和结果文件。

    Args:
        directory: 目录路径
        check_configs: 是否检查配置文件
        check_results: 是否检查结果文件

    Returns:
        检查统计信息
    """
    directory = Path(directory)
    checker = ProtocolChecker(verbose=False)

    stats = {
        'configs_checked': 0,
        'configs_compliant': 0,
        'results_checked': 0,
        'results_compliant': 0,
        'non_compliant_files': []
    }

    # 检查配置文件
    if check_configs:
        for config_file in directory.rglob('*.yaml'):
            if 'unified' in config_file.name or 'official' in config_file.name:
                stats['configs_checked'] += 1
                is_compliant, issues, _ = checker.check_config(str(config_file))
                if is_compliant:
                    stats['configs_compliant'] += 1
                else:
                    stats['non_compliant_files'].append({
                        'file': str(config_file),
                        'type': 'config',
                        'issues': issues
                    })

    # 检查结果文件
    if check_results:
        for result_file in directory.rglob('*result*.json'):
            stats['results_checked'] += 1
            is_compliant, issues, _ = checker.check_result(str(result_file))
            if is_compliant:
                stats['results_compliant'] += 1
            else:
                stats['non_compliant_files'].append({
                    'file': str(result_file),
                    'type': 'result',
                    'issues': issues
                })

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Check protocol compliance for OpenCSLR configs and results'
    )
    parser.add_argument(
        'path',
        help='Path to config file, result file, or directory'
    )
    parser.add_argument(
        '--type',
        choices=['config', 'result', 'auto'],
        default='auto',
        help='Type of file to check (auto-detect by default)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: check all files in directory'
    )
    parser.add_argument(
        '--no-verbose',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    path = Path(args.path)

    if args.batch or path.is_dir():
        # 批量检查
        print(f"Checking directory: {path}")
        stats = check_directory(str(path))

        print("\n" + "=" * 70)
        print("Batch Compliance Check Summary")
        print("=" * 70)
        print(f"Configs checked:    {stats['configs_checked']}")
        print(f"Configs compliant:  {stats['configs_compliant']}")
        print(f"Results checked:    {stats['results_checked']}")
        print(f"Results compliant:  {stats['results_compliant']}")

        if stats['non_compliant_files']:
            print(f"\n❌ Non-compliant files ({len(stats['non_compliant_files'])}):")
            for item in stats['non_compliant_files']:
                print(f"\n  {item['file']} ({item['type']}):")
                for issue in item['issues']:
                    print(f"    - {issue}")
        else:
            print("\n✅ All files are compliant!")

        print("=" * 70 + "\n")

    else:
        # 单文件检查
        checker = ProtocolChecker(verbose=not args.no_verbose)

        if args.type == 'config' or (args.type == 'auto' and path.suffix in ['.yaml', '.yml']):
            is_compliant, _, _ = checker.check_config(str(path))
        elif args.type == 'result' or (args.type == 'auto' and path.suffix == '.json'):
            is_compliant, _, _ = checker.check_result(str(path))
        else:
            print(f"Cannot determine file type for {path}")
            return 1

        return 0 if is_compliant else 1


if __name__ == '__main__':
    exit(main())
