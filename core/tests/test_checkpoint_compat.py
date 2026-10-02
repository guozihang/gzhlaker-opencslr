# -*- coding: utf-8 -*-
"""权重加载的兼容判定(纯标准库,不需要 torch)。

覆盖 ``utils/checkpoint_compat.py`` 的放行规则,以及
``ExperimentManager.load_model_weights`` 确实接上了这套判定 —— 后者用源码断言
检查(该模块顶层 import torch,本环境无法导入)。

    python3 core/tests/test_checkpoint_compat.py
"""

import ast
import sys
import unittest
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from utils.checkpoint_compat import (  # noqa: E402  需先补 sys.path
    ALIAS_CONV1D_PREFIX,
    DEAD_SLOWFAST_FC_PREFIX,
    PRIMARY_CONV1D_PREFIX,
    alias_source_key,
    is_inert_missing,
    is_inert_unexpected,
    partition_key_diff,
)

EXPERIMENT_MANAGER = CORE_ROOT / "manager" / "experiment_manager.py"

# 本地旧权重(04ede88..9bd3e00 期间)相对当前模型的典型缺失键
LEGACY_MISSING = [
    DEAD_SLOWFAST_FC_PREFIX + "0.weight",
    DEAD_SLOWFAST_FC_PREFIX + "0.bias",
    DEAD_SLOWFAST_FC_PREFIX + "2.bias",
    ALIAS_CONV1D_PREFIX + "conv1d.fc.weight",
    ALIAS_CONV1D_PREFIX + "conv1d.temporal_conv.0.weight",
    "spatial_module_container.module_list.0.bn1.num_batches_tracked",
]


class AliasMappingTests(unittest.TestCase):
    def test_alias_maps_to_the_primary_parameter_path(self):
        key = ALIAS_CONV1D_PREFIX + "conv1d.fc.weight"
        self.assertEqual(alias_source_key(key), PRIMARY_CONV1D_PREFIX + "conv1d.fc.weight")

    def test_non_alias_keys_have_no_source(self):
        for key in (PRIMARY_CONV1D_PREFIX + "conv1d.fc.weight",
                    "temporal_module_container.module_list.1.classifier.0.weight",
                    "spatial_module_container.module_list.0.conv1.weight"):
            self.assertIsNone(alias_source_key(key), key)

    def test_alias_keys_are_inert_in_both_directions(self):
        key = ALIAS_CONV1D_PREFIX + "conv1d.temporal_conv.0.weight"
        self.assertTrue(is_inert_missing(key))
        self.assertTrue(is_inert_unexpected(key))


class InertKeyTests(unittest.TestCase):
    def test_dead_outer_fc_is_inert(self):
        self.assertTrue(is_inert_missing(DEAD_SLOWFAST_FC_PREFIX + "1.weight"))
        self.assertTrue(is_inert_unexpected(DEAD_SLOWFAST_FC_PREFIX + "1.bias"))

    def test_batchnorm_counter_missing_is_tolerated_but_extra_is_not(self):
        key = "spatial_module_container.module_list.0.bn1.num_batches_tracked"
        self.assertTrue(is_inert_missing(key))
        self.assertFalse(is_inert_unexpected(key))

    def test_real_parameters_are_never_inert(self):
        for key in ("temporal_module_container.module_list.0.conv1d.conv1d.fc.weight",
                    "spatial_module_container.module_list.0.slow_path.blocks.1.conv1.weight",
                    "temporal_module_container.module_list.1.classifier.0.weight",
                    "loss_module_container.module_list.0.ctc.weight"):
            self.assertFalse(is_inert_missing(key), key)
            self.assertFalse(is_inert_unexpected(key), key)


class PartitionTests(unittest.TestCase):
    def test_legacy_checkpoint_can_be_loaded_directly(self):
        report = partition_key_diff(LEGACY_MISSING, [])
        self.assertEqual(report["fatal_missing"], [])
        self.assertEqual(report["fatal_unexpected"], [])
        self.assertEqual(len(report["ignorable_missing"]), len(LEGACY_MISSING))

    def test_upstream_checkpoint_has_no_diff_at_all(self):
        report = partition_key_diff([], [])
        self.assertEqual(report, {"ignorable_missing": [], "fatal_missing": [],
                                 "ignorable_unexpected": [], "fatal_unexpected": []})

    def test_wrong_fuse_is_fatal(self):
        """FUSE 选错会让慢路径 stage 的通道数变化 —— 必须拒绝,不能悄悄加载。"""
        report = partition_key_diff(
            ["spatial_module_container.module_list.0.slow_path.blocks.1.conv1.weight"], [])
        self.assertEqual(len(report["fatal_missing"]), 1)
        self.assertEqual(report["ignorable_missing"], [])

    def test_extra_real_tensors_are_fatal(self):
        report = partition_key_diff([], ["spatial_module_container.module_list.0.new_block.weight"])
        self.assertEqual(len(report["fatal_unexpected"]), 1)

    def test_report_lists_are_sorted_and_complete(self):
        report = partition_key_diff(LEGACY_MISSING + ["a.real.weight"], [])
        self.assertEqual(report["ignorable_missing"], sorted(report["ignorable_missing"]))
        self.assertEqual(report["fatal_missing"], ["a.real.weight"])


class LoaderWiringTests(unittest.TestCase):
    """load_model_weights 必须真的走这套判定(源码级断言,避免被改回 strict=True)。"""

    def setUp(self):
        self.source = EXPERIMENT_MANAGER.read_text(encoding="utf-8")
        tree = ast.parse(self.source)
        self.func = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_model_weights"
        )

    def test_imports_the_compat_helpers(self):
        self.assertIn("from utils.checkpoint_compat import", self.source)
        for name in ("partition_key_diff", "describe_diff"):
            self.assertIn(name, self.source)

    def test_loads_with_strict_false_then_decides(self):
        call = next(node for node in ast.walk(self.func)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load_state_dict")
        keywords = {kw.arg: kw.value for kw in call.keywords}
        self.assertIn("strict", keywords)
        self.assertIsInstance(keywords["strict"], ast.Constant)
        self.assertFalse(keywords["strict"].value, "必须先非严格加载再自行判定")
        self.assertIn("partition_key_diff", ast.unparse(self.func))

    def test_raises_on_fatal_difference(self):
        source = ast.unparse(self.func)
        self.assertIn("fatal_missing", source)
        self.assertIn("fatal_unexpected", source)
        self.assertIn("RuntimeError", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
