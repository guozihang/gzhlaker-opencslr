# -*- coding: utf-8 -*-
"""把「模型结构与上游 immc-lab/OpenCSLR 对齐」的关键约定钉住(纯标准库)。

背景:本仓库重构后与上游出现过两处真结构差异(SEN 时序卷积、SlowFast FUSE),
另外还有一批死参数/别名键影响 checkpoint 逐 key 对齐。这些都已经对齐,并用本
文件守住 —— 以后再重构时若不小心改回去,这里会直接失败。

判定基准:上游 immc-lab/OpenCSLR(ebdb77f)的 ``core/models`` 与
``core/models/senmodules``。断言方式全部是源码/AST 级(模型模块顶层 import
torch,本环境无法导入)。

    python3 core/tests/test_upstream_structure_alignment.py
"""

import ast
import re
import sys
import unittest
from pathlib import Path

import yaml

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

MODELS_DIR = CORE_ROOT / "models"
MODULES_DIR = CORE_ROOT / "modules"
SLOWFAST_CONFIGS = MODULES_DIR / "spatio" / "slowfast_modules" / "configs"

SLOWFAST_MAIN = SLOWFAST_CONFIGS / "SLOWFAST_64x2_R101_50_50.yaml"
SLOWFAST_FUSE_BIADD = SLOWFAST_CONFIGS / "SLOWFAST_64x2_R101_50_50_FuseBiAdd.yaml"

# 上游 data dict 的键(逐个核对过,见 models/__init__.py 的 Keys)
UPSTREAM_DATA_KEYS = {
    "VID": "vid",
    "VID_LGT": "vid_lgt",
    "LABEL": "label",
    "LABEL_LGT": "label_lgt",
    "INFO": "info",
    "FRAMEWISE_FEATURES": "framewise_features",
    "VISUAL_FEAT": "visual_feat",
    "CONV_LOGITS": "conv_logits",
    "FEAT_LEN": "feat_len",
    "PREDICTIONS": "predictions",
    "SEQUENCE_LOGITS": "sequence_logits",
    "LOSS_LIFTPOOL_U": "loss_LiftPool_u",
    "LOSS_LIFTPOOL_P": "loss_LiftPool_p",
    "HIDDEN": "hidden",
    "LOSS": "loss",
    "TOTAL_LOSS": "total_loss",
    "RECOGNIZED_SENTS": "recognized_sents",
}

# 上游 build_function.py 提供的 5 个模型
UPSTREAM_MODELS = {"slowfast", "tlp", "vac", "corrnet", "sen"}


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def _tree(path):
    return ast.parse(_read(path))


def _registered_models():
    names = set()
    for path in MODELS_DIR.glob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "register_model" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    names.add(arg.value)
    return names


class KeysContractTests(unittest.TestCase):
    """data dict 的键必须与上游字面量逐字符一致 —— 键错一个就是喂错张量。"""

    def test_all_keys_match_upstream_literals(self):
        source = _read(MODELS_DIR / "__init__.py")
        found = dict(re.findall(r'^\s{4}([A-Z_0-9]+)\s*=\s*"([^"]+)"', source, re.M))
        for name, value in UPSTREAM_DATA_KEYS.items():
            with self.subTest(key=name):
                self.assertIn(name, found, "Keys 少了 {}".format(name))
                self.assertEqual(found[name], value)

    def test_no_extra_unmapped_keys(self):
        source = _read(MODELS_DIR / "__init__.py")
        found = set(re.findall(r'^\s{4}([A-Z_0-9]+)\s*=\s*"([^"]+)"', source, re.M))
        extra = {name for name, _ in found} - set(UPSTREAM_DATA_KEYS)
        self.assertEqual(extra, set(), "新增的键也要在上游找到对应来源: {}".format(sorted(extra)))


class ModelRegistryTests(unittest.TestCase):
    def test_every_upstream_model_is_registered(self):
        self.assertEqual(_registered_models(), UPSTREAM_MODELS)

    def test_registry_rejects_unknown_names(self):
        source = _read(MODELS_DIR / "__init__.py")
        self.assertIn("def get_model_builder", source)
        self.assertIn("MODEL_BUILDERS", source)


class SenStructureTests(unittest.TestCase):
    """SEN 的时序卷积必须是上游的 MaxPool 版(默认),LiftPool 只能显式选。"""

    def setUp(self):
        self.sen_source = _read(MODELS_DIR / "sen.py")
        self.sen_module = _read(MODULES_DIR / "temporal" / "SENTemporalConv.py")

    def test_default_temporal_conv_is_upstream_maxpool(self):
        self.assertIn('args.get("temporal_conv", "maxpool")', self.sen_source)
        # 默认分支必须用上游的 sen_TemporalConv,而不是 TLP 的 LiftPool 版
        tree = ast.parse(self.sen_source)
        func = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_build_temporal")
        body = ast.unparse(func)
        self.assertIn("return sen_TemporalConv(args)", body)
        self.assertLess(body.index("sen_TemporalConv(args)"), body.index("TemporalConv1D(args)"),
                        "maxpool 分支要写在 liftpool 分支之前")

    def test_liftpool_remains_available_for_local_weights(self):
        self.assertIn("liftpool", self.sen_source)
        self.assertIn("TemporalConv1D(args)", self.sen_source)

    def test_temporal_conv_uses_maxpool_with_ceil_mode_false(self):
        """上游 'P<n>' 层是 MaxPool1d(ceil_mode=False);LiftPool 版会带可学习参数。"""
        tree = ast.parse(self.sen_module)
        pool_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "MaxPool1d"]
        self.assertTrue(pool_calls, "SENTemporalConv 必须用 MaxPool1d 下采样")
        keywords = {kw.arg: kw.value for kw in pool_calls[0].keywords}
        self.assertIn("ceil_mode", keywords)
        self.assertFalse(keywords["ceil_mode"].value)
        self.assertIn("nn.Conv1d", self.sen_module)
        self.assertIn("nn.BatchNorm1d", self.sen_module)

    def test_conv_type_map_matches_upstream(self):
        # 不能 import(modules/__init__ 会拉 torch),直接从源码 AST 取字面量
        tree = ast.parse(self.sen_module)
        assignment = next(node for node in tree.body
                         if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "CONV_TYPES"
                                 for t in node.targets))
        self.assertEqual(ast.literal_eval(assignment.value),
                         {0: ["K3"], 1: ["K5", "P2"], 2: ["K5", "P2", "K5", "P2"]})

    def test_kernel_size_only_wins_when_conv_type_absent(self):
        """上游语义:配置里有 conv_type 时忽略 kernel_size。"""
        self.assertIn('use_config_kernel = "conv_type" not in args', self.sen_module)

    def test_decoder_skips_decoding_while_training(self):
        """上游 sen_Decoder 只在 eval 解码;训练期跑束搜索纯属浪费(还是纯 Python 实现)。"""
        tree = ast.parse(self.sen_source)
        decoder = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "SENDecoder")
        guards = [node for node in ast.walk(decoder)
                  if isinstance(node, ast.If) and "not self.training" in ast.unparse(node.test)]
        self.assertTrue(guards, "解码必须被 `if not self.training` 包住")
        guarded_body = ast.unparse(guards[0])
        self.assertIn("self.decoder.decode", guarded_body)
        # 而且整段代码里只有这一处解码调用(没有漏在门控之外的第二处)
        calls = [node for node in ast.walk(decoder)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "decode"]
        self.assertEqual(len(calls), 1)
        self.assertIn(calls[0], list(ast.walk(guards[0])), "唯一的解码调用必须在门控内")


class SlowFastStructureTests(unittest.TestCase):
    def test_default_config_matches_upstream_fuse(self):
        config = yaml.safe_load(_read(SLOWFAST_MAIN))
        self.assertEqual(config["SLOWFAST"]["FUSE"], "FuseFastToSlow",
                         "默认必须是上游的 FuseFastToSlow(也是本仓库重构前权重的结构)")

    def test_variant_keeps_the_local_historical_fuse(self):
        config = yaml.safe_load(_read(SLOWFAST_FUSE_BIADD))
        self.assertEqual(config["SLOWFAST"]["FUSE"], "FuseBiAdd")

    def test_variant_differs_from_default_only_in_fuse(self):
        main = yaml.safe_load(_read(SLOWFAST_MAIN))
        variant = yaml.safe_load(_read(SLOWFAST_FUSE_BIADD))
        main["SLOWFAST"].pop("FUSE")
        variant["SLOWFAST"].pop("FUSE")
        self.assertEqual(main, variant, "变体只应该差 FUSE 一行")

    def test_network_yaml_points_at_the_upstream_config_by_default(self):
        network = yaml.safe_load(_read(CORE_ROOT / "configs" / "network.yaml"))
        self.assertEqual(
            network["slowfast"]["model_args"]["slowfast_config"],
            "SLOWFAST_64x2_R101_50_50.yaml",
        )

    def test_dead_outer_fc_is_kept_for_upstream_state_dict_parity(self):
        """断言要精确到包装类的 __init__:同名赋值在 TemporalSlowFastFuse 里也有一处。"""
        tree = ast.parse(_read(MODULES_DIR / "temporal" / "TemporalSlowFastConv1D.py"))
        wrapper = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "TemporalSlowFastConv1D")
        init = next(node for node in wrapper.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assigned = set()
        for node in ast.walk(init):
            for target in getattr(node, "targets", []):
                if (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"):
                    assigned.add(target.attr)
        self.assertIn("conv1d", assigned)
        self.assertIn("fc", assigned,
                      "上游 checkpoint 带着 TemporalSlowFastConv1D 外层从不使用的 fc,"
                      "删掉就无法与上游权重逐 key 对齐")

    def test_temporal_model_registers_conv1d_like_upstream(self):
        source = _read(MODULES_DIR / "temporal" / "temporal_model.py")
        self.assertIn("self.conv1d = conv1d", source)
        self.assertNotIn("object.__setattr__", source,
                         "本地曾用它避开重复注册,但那会让 state_dict 少一组别名键")


class SharedPrimitiveTests(unittest.TestCase):
    """共用积木里被上游复用的那些关键算子,别在重构中被换掉。"""

    def test_liftpool_is_still_the_tlp_pooling(self):
        source = _read(MODULES_DIR / "others" / "liftpool.py")
        self.assertIn("class TemporalLiftPooling", source)
        self.assertIn("class Local_Weighting", source)

    def test_vac_temporal_conv_uses_maxpool(self):
        """VAC 一直用 MaxPool 版(上游 VACTemporalConv 同款),别被改成 LiftPool。"""
        source = _read(MODULES_DIR / "temporal" / "tconv.py")
        self.assertIn("class VACTemporalConv", source)
        self.assertIn("nn.MaxPool1d", source)

    def test_tcn_based_models_use_conv_type_tables(self):
        for name in ("corrnet_tconv.py", "TemporalSlowFastConv1D.py"):
            with self.subTest(module=name):
                source = _read(MODULES_DIR / "temporal" / name)
                self.assertIn("conv_type", source)
                self.assertIn("kernel_size", source)

    def test_config_whitelist_accepts_the_alignment_switches(self):
        from manager.config_manager import ConfigManager
        model_args = ConfigManager.KNOWN_NESTED_KEYS["model_args"]
        for key in ("temporal_conv", "input_size"):
            self.assertIn(key, model_args, "新开关不写进白名单会被当成拼写错误拒掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
