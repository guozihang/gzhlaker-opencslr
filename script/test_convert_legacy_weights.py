#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_legacy_weights.py 的单元测试（CPU，无需下载预训练权重）。

用法（仓库根目录执行）::

    python script/test_convert_legacy_weights.py

也可直接 pytest 跑。测试覆盖三件事：

1. 旧 checkpoint 的两种包装（训练 checkpoint / 裸 state_dict）与 ``module.`` 前缀
   都能被正确拆开；
2. ``analyze`` 能把多出来的 key 分对类——与保留 key 共享 storage 的算重复注册、
   命中已知死模块前缀的算可丢弃、其余进 orphan 阻断转换，形状不一致也能报出来；
3. ``convert`` 走完整流程后，写出的 checkpoint 能被当前模型 strict 加载，且每个保留
   张量与输入逐元素相同。
"""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

import convert_legacy_weights as clw  # noqa: E402

MODEL = "tlp"


def _build():
    _, cfg = clw.resolve_config(None, MODEL)
    return clw.build_model(MODEL, cfg)


def test_read_checkpoint_unwraps_wrapper_and_module_prefix():
    """训练 checkpoint 的包装与 DataParallel 的 module. 前缀都要被剥掉。"""
    model = _build()
    with tempfile.TemporaryDirectory() as tmp:
        payload = {
            "epoch": 3,
            "model_state_dict": {"module." + k: v for k, v in model.state_dict().items()},
            "optimizer_state_dict": {},
        }
        path = Path(tmp) / "wrapped.pt"
        torch.save(payload, path)

        meta, state = clw.read_checkpoint(str(path))
        assert meta is not None and meta["epoch"] == 3
        assert set(state) == set(model.state_dict())
        assert not any(k.startswith("module.") for k in state)

        raw = Path(tmp) / "raw.pt"
        torch.save(model.state_dict(), raw)
        meta, state = clw.read_checkpoint(str(raw))
        assert meta is None
        assert set(state) == set(model.state_dict())
    print("[OK] read_checkpoint 拆包与 module. 前缀清理")


def test_read_checkpoint_rejects_unknown_format():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.pt"
        torch.save({"foo": "bar"}, path)
        try:
            clw.read_checkpoint(str(path))
        except SystemExit as exc:
            assert "无法识别" in str(exc)
        else:
            raise AssertionError("未知格式应当报错退出")
    print("[OK] read_checkpoint 对未知格式报错")


def test_analyze_classifies_extra_keys_and_shapes():
    model = _build()
    sd = dict(model.state_dict())
    some_key = sorted(sd)[0]

    alias = "temporal_module_container.module_list.9.conv1d.conv1d.dup.weight"
    sd[alias] = sd[some_key]                       # 与保留 key 共享 storage
    sd["temporal_module_container.module_list.0.fc.0.weight"] = torch.zeros(1)  # 已知死模块
    sd["totally.unknown.weight"] = torch.zeros(1)  # 无法解释
    del sd[sorted(sd)[1]]                          # 制造一个缺 key

    diff = clw.analyze(sd, model)
    assert alias in diff["duplicate_extra"], diff["duplicate_extra"]
    assert "temporal_module_container.module_list.0.fc.0.weight" in diff["dead_extra"]
    assert "totally.unknown.weight" in diff["orphan_extra"]
    assert len(diff["missing"]) == 1
    assert diff["shape_mismatch"] == []

    broken = dict(model.state_dict())
    broken[some_key] = torch.zeros(1)
    assert [m[0] for m in clw.analyze(broken, model)["shape_mismatch"]] == [some_key]
    print("[OK] analyze 分类：重复注册 / 死模块 / orphan / 形状不一致")


def test_convert_roundtrip_is_lossless_and_loads_strictly():
    model = _build()
    target = model.state_dict()
    sd = dict(target)
    # 旧代码的两种遗留 key 都塞进来，转换后应当被丢掉
    sd["temporal_module_container.module_list.9.conv1d.conv1d.dup.weight"] = sd[sorted(sd)[0]]
    sd["temporal_module_container.module_list.0.fc.0.weight"] = torch.zeros(1)

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "legacy.pt"
        dst = Path(tmp) / "converted.pt"
        torch.save({"epoch": 7, "model_state_dict": sd,
                    "optimizer_state_dict": {"x": 1}}, src)

        args = Namespace(checkpoint=str(src), out=str(dst), model=MODEL, exp=None,
                         model_arg=None, dry_run=False, inference_only=True)
        assert clw.cmd_convert(args) == 0
        assert dst.is_file()

        out = torch.load(dst, map_location="cpu", weights_only=False)
        assert out["epoch"] == 7
        assert "optimizer_state_dict" not in out      # --inference-only
        assert set(out["model_state_dict"]) == set(target)
        assert "temporal_module_container.module_list.0.fc.0.weight" not in out["model_state_dict"]

        model.load_state_dict(out["model_state_dict"], strict=True)
        for key, value in out["model_state_dict"].items():
            assert torch.equal(value, target[key]), key
    print("[OK] convert 往返：strict 加载通过且逐元素相同")


def test_detect_sen_temporal_reads_key_depth():
    """SEN 两种时序卷积靠 key 的层级区分：旧 MaxPool 版是摊平的 Sequential。"""
    assert clw.detect_sen_temporal(
        {"temporal_module_container.module_list.0.conv1d.temporal_conv.3.weight": 1}) == "maxpool"
    assert clw.detect_sen_temporal(
        {"temporal_module_container.module_list.0.conv1d.temporal_conv.1.predictor.0.weight": 1}
    ) == "liftpool"
    assert clw.detect_sen_temporal({"temporal_module_container.module_list.0.conv1d.fc.weight": 1}) is None
    print("[OK] detect_sen_temporal 判别两种 SEN 时序卷积")


def test_sen_legacy_checkpoint_is_rebuilt_with_maxpool_temporal():
    """旧 SEN checkpoint：默认(LiftPool)结构建不出来，脚本要自动切到 MaxPool 版。"""
    cfg = {
        "model": "sen",
        "model_args": {"num_classes": 8, "conv_type": 2, "use_bn": 1,
                       "kernel_size": ["K5", "P2", "K5", "P2"], "stride": [4, 0]},
        "loss_weights": {"ConvCTC": 1.0, "SeqCTC": 1.0},
    }
    legacy_args = dict(cfg["model_args"], temporal_conv="maxpool")
    state = dict(clw.build_model("sen", dict(cfg, model_args=legacy_args)).state_dict())
    assert clw.detect_sen_temporal(state) == "maxpool"

    default_model = clw.build_model("sen", cfg)
    assert clw.detect_sen_temporal(default_model.state_dict()) == "liftpool"
    diff = clw.analyze(state, default_model)
    assert diff["missing"], "LiftPool 版应该缺掉旧 MaxPool 版时序卷积的 key"
    assert diff["shape_mismatch"] == []
    # 旧时序卷积的 key 在当前模型里解释不了，analyze 会因为这一条阻断自动转换
    assert diff["orphan_extra"], "旧 MaxPool 版的时序卷积 key 应被报成无法解释"

    rebuilt = clw.build_model_for("sen", cfg, None, state)
    assert set(rebuilt.state_dict()) == set(state)
    rebuilt.load_state_dict(state, strict=True)

    # 显式指定的实现不该被自动改掉
    explicit = clw.build_model_for("sen", cfg, {"temporal_conv": "liftpool"}, state)
    assert set(explicit.state_dict()) == set(default_model.state_dict())
    print("[OK] 旧 SEN checkpoint 自动按 temporal_conv=maxpool 重建并 strict 加载")


if __name__ == "__main__":
    test_read_checkpoint_unwraps_wrapper_and_module_prefix()
    test_read_checkpoint_rejects_unknown_format()
    test_analyze_classifies_extra_keys_and_shapes()
    test_convert_roundtrip_is_lossless_and_loads_strictly()
    test_detect_sen_temporal_reads_key_depth()
    test_sen_legacy_checkpoint_is_rebuilt_with_maxpool_temporal()
    print("\nAll convert_legacy_weights tests passed.")
