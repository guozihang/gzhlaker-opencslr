# -*- encoding: utf-8 -*-
"""动作验证器。

在执行前验证所有 Agent 提议的动作，确保：
1. JSON schema 合法
2. 组件名在注册表中存在
3. 文件路径有效
4. Seed 在白名单中
5. 预算充足
"""

import json
from typing import Dict, Any, List
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaError

from .budget import BudgetGuard, validate_seed, ALLOWED_SEEDS


class ValidationError(Exception):
    """验证错误。"""
    pass


# 动作 Schema 定义
ACTION_SCHEMAS = {
    "evaluate_decoder": {
        "type": "object",
        "required": ["action", "checkpoint", "dataset", "split", "decoder_config"],
        "properties": {
            "action": {"const": "evaluate_decoder"},
            "checkpoint": {"type": "string", "minLength": 1},
            "dataset": {
                "enum": ["phoenix2014", "phoenix2014t", "csl-daily"]
            },
            "split": {"enum": ["dev", "test"]},
            "decoder_config": {
                "type": "object",
                "properties": {
                    "decode_mode": {"enum": ["greedy", "beam"]},
                    "beam_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20
                    },
                    "lm_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0
                    }
                },
                "required": ["decode_mode"]
            },
            "hypothesis": {"type": "string"},
            "estimated_time_minutes": {"type": "number", "minimum": 0}
        }
    },

    "generate_config": {
        "type": "object",
        "required": ["action", "base_config", "overrides", "output_path"],
        "properties": {
            "action": {"const": "generate_config"},
            "base_config": {"type": "string"},
            "overrides": {"type": "object"},
            "output_path": {"type": "string"},
            "validate": {"type": "boolean"}
        }
    },

    "submit_training": {
        "type": "object",
        "required": ["action", "config_path", "estimated_gpu_hours"],
        "properties": {
            "action": {"const": "submit_training"},
            "config_path": {"type": "string"},
            "estimated_gpu_hours": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 10.0
            },
            "gpus": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0}
            },
            "use_watchdog": {"type": "boolean"},
            "rationale": {"type": "string"}
        }
    }
}


class ActionValidator:
    """动作验证器。

    在动作执行前进行多层验证。
    """

    def __init__(self,
                 budget_guard: BudgetGuard,
                 check_registry: bool = True,
                 check_files: bool = True):
        """初始化验证器。

        Args:
            budget_guard: 预算守卫实例
            check_registry: 是否检查注册表
            check_files: 是否检查文件存在性
        """
        self.budget_guard = budget_guard
        self.check_registry = check_registry
        self.check_files = check_files

        # 加载注册表（如果需要）
        self.registries = {}
        if check_registry:
            self._load_registries()

    def _load_registries(self):
        """加载组件注册表。"""
        try:
            import sys
            sys.path.insert(0, "./core")
            from registry import (
                MODEL_REGISTRY,
                BACKBONE_REGISTRY,
                TEMPORAL_REGISTRY,
                DECODER_REGISTRY,
                DATASET_REGISTRY
            )

            self.registries = {
                "model": MODEL_REGISTRY,
                "backbone": BACKBONE_REGISTRY,
                "temporal": TEMPORAL_REGISTRY,
                "decoder": DECODER_REGISTRY,
                "dataset": DATASET_REGISTRY
            }
        except ImportError as e:
            print(f"Warning: Could not load registries: {e}")
            self.check_registry = False

    def validate(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """验证动作。

        Args:
            action: 动作字典

        Returns:
            验证结果

        Raises:
            ValidationError: 验证失败时抛出
        """
        errors = []
        warnings = []

        # 1. 检查动作类型
        action_type = action.get("action")
        if not action_type:
            raise ValidationError("Missing 'action' field")

        if action_type not in ACTION_SCHEMAS:
            raise ValidationError(
                f"Unknown action type: {action_type}. "
                f"Available: {list(ACTION_SCHEMAS.keys())}"
            )

        # 2. JSON Schema 验证
        try:
            validate(instance=action, schema=ACTION_SCHEMAS[action_type])
        except JsonSchemaError as e:
            errors.append(f"Schema validation failed: {e.message}")

        # 3. 特定动作的额外验证
        if action_type == "evaluate_decoder":
            self._validate_evaluate_decoder(action, errors, warnings)
        elif action_type == "submit_training":
            self._validate_submit_training(action, errors, warnings)
        elif action_type == "generate_config":
            self._validate_generate_config(action, errors, warnings)

        # 4. 汇总结果
        if errors:
            raise ValidationError(
                f"Action validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return {
            "is_valid": True,
            "errors": errors,
            "warnings": warnings,
            "action_type": action_type
        }

    def _validate_evaluate_decoder(self,
                                   action: Dict[str, Any],
                                   errors: List[str],
                                   warnings: List[str]):
        """验证 evaluate_decoder 动作。"""
        # 检查 checkpoint 文件
        checkpoint = action.get("checkpoint")
        if self.check_files and checkpoint:
            if not Path(checkpoint).exists():
                errors.append(f"Checkpoint not found: {checkpoint}")

        # 检查 dataset
        dataset = action.get("dataset")
        if self.check_registry and dataset:
            if "dataset" in self.registries:
                if not self.registries["dataset"].is_registered(dataset):
                    available = self.registries["dataset"].list_available()
                    errors.append(
                        f"Dataset '{dataset}' not registered. "
                        f"Available: {available}"
                    )

        # 检查 split（test 只能用一次的警告）
        split = action.get("split")
        if split == "test":
            warnings.append(
                "Using test split. Remember: test should only be evaluated once per config."
            )

        # 检查 decoder_config
        decoder_config = action.get("decoder_config", {})
        if decoder_config.get("decode_mode") == "beam":
            if "beam_size" not in decoder_config:
                warnings.append("beam mode without beam_size, will use default")

    def _validate_submit_training(self,
                                  action: Dict[str, Any],
                                  errors: List[str],
                                  warnings: List[str]):
        """验证 submit_training 动作。"""
        # 检查配置文件
        config_path = action.get("config_path")
        if self.check_files and config_path:
            if not Path(config_path).exists():
                errors.append(f"Config not found: {config_path}")

        # 检查 GPU 预算
        estimated_gpu_hours = action.get("estimated_gpu_hours", 0)
        try:
            self.budget_guard.check_gpu_budget(estimated_gpu_hours)
        except Exception as e:
            errors.append(str(e))

        # 检查 GPU 设备
        gpus = action.get("gpus", [])
        if len(gpus) > 4:
            warnings.append(
                f"Using {len(gpus)} GPUs may not be efficient. "
                "Consider 1-2 GPUs for most models."
            )

    def _validate_generate_config(self,
                                  action: Dict[str, Any],
                                  errors: List[str],
                                  warnings: List[str]):
        """验证 generate_config 动作。"""
        # 检查基础配置
        base_config = action.get("base_config")
        if self.check_files and base_config:
            base_path = Path(base_config)
            if not base_path.exists():
                # 尝试在 core/configs/ 下查找
                alt_path = Path("core/configs") / base_config
                if not alt_path.exists():
                    errors.append(f"Base config not found: {base_config}")

        # 检查覆盖项中的组件名
        overrides = action.get("overrides", {})
        if self.check_registry:
            for key in ["model", "backbone", "temporal_module", "decoder"]:
                if key in overrides:
                    component_name = overrides[key]
                    registry_key = key if key in self.registries else key.replace("_module", "")

                    if registry_key in self.registries:
                        registry = self.registries[registry_key]
                        if not registry.is_registered(component_name):
                            available = registry.list_available()
                            errors.append(
                                f"{key} '{component_name}' not registered. "
                                f"Available: {available}"
                            )

        # 检查 seed
        if "random_seed" in overrides:
            seed = overrides["random_seed"]
            try:
                validate_seed(seed)
            except ValueError as e:
                errors.append(str(e))

    def validate_batch(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量验证多个动作。

        Args:
            actions: 动作列表

        Returns:
            批量验证结果
        """
        results = []
        total_gpu_hours = 0.0
        total_api_cost = 0.0

        for i, action in enumerate(actions):
            try:
                result = self.validate(action)
                results.append({
                    "index": i,
                    "status": "valid",
                    "action": action.get("action"),
                    **result
                })

                # 累计资源
                if "estimated_gpu_hours" in action:
                    total_gpu_hours += action["estimated_gpu_hours"]
                if "estimated_api_cost" in action:
                    total_api_cost += action["estimated_api_cost"]

            except ValidationError as e:
                results.append({
                    "index": i,
                    "status": "invalid",
                    "action": action.get("action"),
                    "error": str(e)
                })

        # 检查总预算
        budget_errors = []
        try:
            self.budget_guard.check_gpu_budget(total_gpu_hours)
        except Exception as e:
            budget_errors.append(str(e))

        try:
            self.budget_guard.check_api_budget(total_api_cost)
        except Exception as e:
            budget_errors.append(str(e))

        return {
            "total_actions": len(actions),
            "valid_actions": sum(1 for r in results if r["status"] == "valid"),
            "invalid_actions": sum(1 for r in results if r["status"] == "invalid"),
            "results": results,
            "total_gpu_hours": total_gpu_hours,
            "total_api_cost": total_api_cost,
            "budget_errors": budget_errors
        }


def quick_validate(action: Dict[str, Any], budget_guard: BudgetGuard) -> bool:
    """快速验证（用于简单检查）。

    Args:
        action: 动作字典
        budget_guard: 预算守卫

    Returns:
        是否有效
    """
    validator = ActionValidator(budget_guard, check_registry=False, check_files=False)
    try:
        validator.validate(action)
        return True
    except ValidationError:
        return False
