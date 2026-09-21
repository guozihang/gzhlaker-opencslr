#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""旧权重适配脚本：校验旧 checkpoint 与当前代码的模型结构，并把它转成当前格式。

背景
----
b46e781 之后模型代码做过一次大重构（按 spatio/temporal/losses/decoders/others 分层，
模型文件各占一个）。绝大多数模块只是换了文件位置，**state_dict 逐 key 不变**；但对比
ebdb77f（重构前的参考提交）后有两处结构确实变了，旧 checkpoint 直接
``load_state_dict(strict=True)`` 会失败：

1. SlowFast。旧代码把同一个时序卷积模块注册了两遍
   （``temporal_model.conv1d = conv1d``，于是同一份参数在 state_dict 里出现两次），
   并在 TemporalSlowFastConv1D 上留了一个前向根本没用的 ``fc``。新代码删掉了这两处，
   因此旧 checkpoint 会多出 57 个**冗余 key**：它们要么与保留 key 共享同一块 storage
   （同一个模块的重复注册），要么属于那个死 fc。按当前模型的 key 集合过滤、丢掉多出来的
   部分即可，不损失任何有效权重。

   另外 SlowFast 的融合方式 ``FuseFastToSlow`` / ``FuseBiAdd`` 两种实现在
   fuse_helper.py 里都还在，重构改的只是 SLOWFAST_64x2_R101_50_50.yaml 里的 FUSE 默认值
   （FuseFastToSlow → FuseBiAdd）。两者参数集合不同（BiAdd 多
   ``weight_s``/``weight_f``/``conv_s2f``/``bn2``，FastToSlow 是 ``conv_f2s``+``bn``），
   不是改 key 能解决的——脚本会从 checkpoint 的 key 反推它属于哪一种，和当前配置不一致时
   明确提示换配置。旧权重是 FuseFastToSlow，对应的 yaml 已经放好一份副本
   （``configs/SLOWFAST_64x2_R101_50_50_FuseFastToSlow.yaml``，只差 FUSE 一行），加一个
   ``--model-arg slowfast_config=SLOWFAST_64x2_R101_50_50_FuseFastToSlow.yaml`` 即可。

2. SEN。旧代码用 MaxPool 版时序卷积（senmodules/sen_TemporalConv.py），重构后 build_sen
   默认换成 LiftPool 版 TemporalConv，参数集合不同（LiftPool 版多出 predictor/updater/
   weight1/weight2）。这一处现在也能无损转换：``model_args.temporal_conv`` 设为
   ``maxpool`` 时 build_sen 按旧结构建模型（复用等价的 VACTemporalConv，key 与旧代码逐字
   相同）。脚本会从 checkpoint 的 key 形状判断出来并自动切过去，network.yaml 的 sen 节
   默认也是 ``maxpool``。

TLP / VAC / CorrNet 的 state_dict 与 ebdb77f 逐 key 完全一致（名称、形状、共享关系都
相同），脚本只做 wrapper 拆包和 ``module.`` 前缀清理。

用法
----
::

    # 只校验（不写文件）：打印 checkpoint 与当前模型结构的差异
    python script/convert_legacy_weights.py check --model slowfast --checkpoint old.pt

    # 转换，并用当前模型 strict 加载验证
    python script/convert_legacy_weights.py convert --model slowfast \
        --checkpoint old.pt --out new.pt

    # 走某个实验的配置（实验节按 ConfigManager 的规则覆盖网络节）
    python script/convert_legacy_weights.py convert --exp baseline \
        --checkpoint old.pt --out new.pt

退出码：0 = 可以转换/已转换，1 = 有无法自动处理的结构差异，2 = 参数或输入有问题。
"""

import argparse
import contextlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
CORE = REPO_ROOT / "core"
# core 在 sys.path 上,才能 `from models import ...` / `from modules import ...`
sys.path.insert(0, str(CORE))
# 复用 dump_model_structures 的配置读取(实验节覆盖网络节,与 ConfigManager 一致)
sys.path.insert(0, str(SCRIPT_DIR))

import torch  # noqa: E402

from dump_model_structures import load_sections  # noqa: E402

# ResNet(args) 写死了 torchvision 的 ImageNet 预训练权重；corrnet_resnet / senresnet
# 通过 model_zoo.load_url 拉 resnet18。转换只需要键名和形状，跳过这些下载。
_TORCHVISION_BACKBONES = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "mobilenet_v2", "squeezenet1_1", "shufflenet_v2_x1_0", "efficientnet_b1",
    "mnasnet1_0", "regnet_y_800mf", "vgg16_bn", "vgg11_bn", "regnet_x_800mf",
    "regnet_x_400mf", "densenet121", "regnet_y_1_6gf",
)

# SlowFast 融合方式的判别特征：FuseBiAdd 有 conv_s2f/bn2/weight_s/weight_f，
# FuseFastToSlow 只有 conv_f2s+bn。
_FUSE_KEYS = {
    "FuseBiAdd": ".conv_s2f.weight",
    "FuseFastToSlow": ".conv_f2s.weight",
}

# ebdb77f 的 TemporalSlowFastConv1D 上有一个前向从未使用的 fc（旧 forward 只调用
# self.conv1d(...)），它在 state_dict 里通过两条路径各出现一次：
# module_list.0.fc.* 和 module_list.1.conv1d.fc.*（后者来自 temporal_model.conv1d 的
# 重复注册）。新代码把这个死 fc 删掉了，丢弃它不损失任何前向会用到的权重。
_LEGACY_DEAD_PREFIXES = (
    "temporal_module_container.module_list.0.fc.",
)

# SEN 时序卷积的两种结构：旧 MaxPool 版把各层摊平成一个 nn.Sequential
# （temporal_conv.<i>.weight），当前 LiftPool 版是嵌套的 ModuleList
# （temporal_conv.<i>.<j>.weight）。key 的层数就是判别特征。
_SEN_TEMPORAL_MARKER = ".conv1d.temporal_conv."
_SEN_MAXPOOL = "maxpool"
_SEN_LIFTPOOL = "liftpool"


@contextlib.contextmanager
def skip_pretrained():
    """构造模型时跳过预训练权重的下载/加载（只影响初始化值，不影响 key 与形状）。"""
    import torchvision.models as tvm
    import torch.utils.model_zoo as model_zoo

    saved_ctor = {}
    for name in _TORCHVISION_BACKBONES:
        orig = getattr(tvm, name, None)
        if orig is None:
            continue
        saved_ctor[name] = orig

        def _make(orig_fn):
            def _ctor(*args, **kwargs):
                kwargs["weights"] = None
                return orig_fn(*args, **kwargs)

            return _ctor

        setattr(tvm, name, _make(orig))

    saved_load_url = model_zoo.load_url
    model_zoo.load_url = lambda *a, **kw: {}
    try:
        yield
    finally:
        model_zoo.load_url = saved_load_url
        for name, orig in saved_ctor.items():
            setattr(tvm, name, orig)


def resolve_config(exp_name, model_name):
    """返回 (网络名, 配置字典)。--exp 优先，否则按 --model 取 network.yaml 的同名节。"""
    if exp_name:
        sections = load_sections(exp_name)
        name = next(iter(sections))
        return name, sections[name]
    if not model_name:
        raise SystemExit("必须给出 --model 或 --exp 之一")
    sections = load_sections(None)
    if model_name not in sections:
        raise SystemExit("network.yaml 里没有网络 {!r}；可用: {}".format(
            model_name, sorted(sections)))
    return model_name, sections[model_name]


def parse_model_args(pairs):
    """把 --model-arg KEY=VALUE 解析成覆盖 model_args 的字典（值按 YAML 解析）。"""
    import yaml

    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit("--model-arg 需要 KEY=VALUE 形式，收到 {!r}".format(pair))
        key, _, raw = pair.partition("=")
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def build_model(net_name, cfg, overrides=None):
    """按当前代码构建模型（CPU，不下载预训练权重）。"""
    from models import get_model_builder

    model_args = dict(cfg.get("model_args") or {})
    model_args.update(overrides or {})
    if "num_classes" not in model_args:
        raise SystemExit("配置 {} 缺少 model_args.num_classes".format(net_name))
    num_classes = int(model_args["num_classes"])
    # 假词表只喂给解码器，不产生参数，不影响 state_dict
    gloss_dict = {i: ("gloss%d" % i,) for i in range(num_classes - 1)}
    builder = get_model_builder(cfg.get("model", net_name))
    with skip_pretrained():
        return builder(model_args, gloss_dict=gloss_dict,
                       loss_weights=cfg.get("loss_weights") or {})


def read_checkpoint(path):
    """读取 checkpoint，返回 (原始容器或 None, 去前缀后的 model_state_dict)。"""
    if not Path(path).is_file():
        raise SystemExit("找不到 checkpoint: {}".format(path))
    obj = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and "model_state_dict" in obj:
        meta, state = obj, obj["model_state_dict"]
    elif isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        meta, state = None, obj
    else:
        raise SystemExit(
            "无法识别的 checkpoint 格式：{}（既不是 state_dict，也没有 model_state_dict）"
            .format(type(obj).__name__))

    normalized = {}
    for key, value in state.items():
        name = key[len("module."):] if key.startswith("module.") else key
        normalized[name.replace(".module.", ".")] = value
    return meta, normalized


def detect_fuse(state):
    """从 state_dict 的 key 反推 SlowFast 的融合实现；不是 SlowFast 时返回 None。"""
    for fuse, marker in _FUSE_KEYS.items():
        if any(key.endswith(marker) for key in state):
            return fuse
    return None


def detect_sen_temporal(state):
    """从 state_dict 的 key 反推 SEN 用的时序卷积实现；判不出来返回 None。

    旧 MaxPool 版是 nn.Sequential，key 形如 ``conv1d.temporal_conv.3.weight``；
    当前 LiftPool 版是 ModuleList，套了一层 nn.Sequential，形如
    ``conv1d.temporal_conv.1.predictor.0.weight``。后者层级更深。
    """
    for key in state:
        if _SEN_TEMPORAL_MARKER not in key:
            continue
        depth = len(key.split(_SEN_TEMPORAL_MARKER, 1)[1].split("."))
        if depth >= 3:
            return _SEN_LIFTPOOL
        if depth == 2:
            return _SEN_MAXPOOL
    return None


def build_model_for(net_name, cfg, overrides, state):
    """按当前代码构建模型；checkpoint 是旧结构时自动补上对应的兼容开关。

    目前只有 SEN 需要：checkpoint 是旧 MaxPool 版时序卷积时，若配置里没显式指定
    temporal_conv，就按 maxpool 重建一次，免得用户还要自己去翻脚本说明。
    """
    model = build_model(net_name, cfg, overrides)
    if detect_sen_temporal(state) != _SEN_MAXPOOL:
        return model
    if cfg.get("model", net_name) != "sen":
        return model
    effective = dict(cfg.get("model_args") or {})
    effective.update(overrides or {})
    if effective.get("temporal_conv") is not None:
        return model                     # 用户显式指定了实现，不擅自改
    print("checkpoint 用的是旧版 MaxPool 时序卷积，按 SEN 旧结构重建模型"
          "（temporal_conv=maxpool）……")
    overrides = dict(overrides or {})
    overrides["temporal_conv"] = _SEN_MAXPOOL
    return build_model(net_name, cfg, overrides)


def analyze(state, model):
    """对比 checkpoint 与当前模型的 key 集合，并把多出来的 key 分成三类。

    Returns:
        dict:
            missing            当前模型需要、checkpoint 里没有的 key；
            shape_mismatch     同名但形状不同的 (key, checkpoint 形状, 当前形状)；
            duplicate_extra    多出来但与某个保留 key 共享 storage（同一模块被重复注册）；
            dead_extra         多出来且命中已知的「旧代码里前向没用到」的模块；
            orphan_extra       多出来且无法解释的，需要人工确认。
    只有 missing / shape_mismatch / orphan_extra 全空时才能自动转换。
    """
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    extra = sorted(set(state) - set(target))
    shape_mismatch = [(key, list(state[key].shape), list(target[key].shape))
                      for key in sorted(set(target) & set(state))
                      if list(state[key].shape) != list(target[key].shape)]

    def storage(tensor):
        return tensor.untyped_storage().data_ptr()

    kept_storages = {storage(state[key]) for key in state if key in target}

    dead = {key for key in extra if key.startswith(_LEGACY_DEAD_PREFIXES)}
    dead_storages = {storage(state[key]) for key in dead}

    duplicate, dead_extra, orphan = [], [], []
    for key in extra:
        if key in dead:
            dead_extra.append(key)
        elif storage(state[key]) in kept_storages:
            duplicate.append(key)
        elif storage(state[key]) in dead_storages:
            # 与已知死模块共享 storage：同一条重复注册路径
            dead_extra.append(key)
        else:
            orphan.append(key)
    return {"missing": missing, "shape_mismatch": shape_mismatch,
            "duplicate_extra": duplicate, "dead_extra": dead_extra,
            "orphan_extra": orphan}


def print_report(net_name, path, meta, state, model, diff, fuse):
    """打印校验报告，返回 True 表示可以无损转换。"""
    target = model.state_dict()
    print("checkpoint : {}".format(path))
    print("包装格式   : {}".format(
        "训练 checkpoint（model_state_dict/epoch/optimizer…）" if meta else "裸 state_dict"))
    print("网络       : {}（当前代码构建出 {} 个张量）".format(net_name, len(target)))
    print("checkpoint : {} 个张量".format(len(state)))

    if fuse is not None:
        current = detect_fuse(target)
        print("SlowFast 融合: checkpoint = {}，当前配置 = {}".format(fuse, current))
        if current != fuse:
            print("  ✗ 融合方式不一致，这不是改 key 能解决的：两种实现（FuseFastToSlow /")
            print("    FuseBiAdd）都还在 fuse_helper.py 里，把 network.yaml 的")
            print("    slowfast_config 指向的 yaml 中 FUSE 一行改成 {} 后重试。"
                  .format(fuse))
            if fuse == "FuseFastToSlow":
                print("    现成的一份副本：configs/SLOWFAST_64x2_R101_50_50_FuseFastToSlow.yaml，")
                print("    用 --model-arg slowfast_config=SLOWFAST_64x2_R101_50_50_FuseFastToSlow.yaml 即可。")

    ok = True
    if diff["missing"]:
        ok = False
        print("\n✗ 当前模型需要、checkpoint 里没有的 key（{} 个）：".format(len(diff["missing"])))
        for key in diff["missing"][:20]:
            print("    - {} {}".format(key, list(target[key].shape)))
        if len(diff["missing"]) > 20:
            print("    ... 其余 {} 个".format(len(diff["missing"]) - 20))
        print("  这些参数只能重新训练；若只是配置和训练时不一致，按下面的提示改配置再跑。")
        if fuse is not None and detect_fuse(target) != fuse:
            print("  SlowFast 融合方式不一致也会表现成大量缺 key，先按上面的提示改配置。")
        sen_style = detect_sen_temporal(state)
        if sen_style is not None and sen_style != detect_sen_temporal(target):
            print("  该 checkpoint 的 SEN 时序卷积是 {} 版，和当前模型（{} 版）不同："
                  .format(sen_style, detect_sen_temporal(target) or "未知"))
            print("  用 --model-arg temporal_conv={} 再跑一次即可（network.yaml 的 sen 节"
                  "默认就是 maxpool）。".format(sen_style))

    if diff["shape_mismatch"]:
        ok = False
        print("\n✗ 同名但形状不同的 key（{} 个）：".format(len(diff["shape_mismatch"])))
        for key, got, want in diff["shape_mismatch"][:20]:
            print("    - {} {} -> {}".format(key, got, want))
        if len(diff["shape_mismatch"]) > 20:
            print("    ... 其余 {} 个".format(len(diff["shape_mismatch"]) - 20))
        print("  形状由配置决定（num_classes / hidden_size / c2d_type / kernel_size 等），")
        print("  说明当前配置和训练时用的配置不一致，请先用训练时的配置再跑一次。")

    for label, keys, why in (
            ("冗余 key", diff["duplicate_extra"], "与保留 key 共享同一块 storage，丢弃不损失权重"),
            ("旧代码遗留的死模块", diff["dead_extra"], "旧前向从未使用，丢弃不损失权重")):
        if not keys:
            continue
        print("\n✓ {}（{} 个，{}）：".format(label, len(keys), why))
        # 按「模块路径」归类（去掉末尾的参数名两段），一眼看出是哪个模块
        groups = {}
        for key in keys:
            parts = key.split(".")
            prefix = ".".join(parts[:-2]) if len(parts) > 2 else key
            groups[prefix] = groups.get(prefix, 0) + 1
        for prefix, count in sorted(groups.items()):
            print("    {} （{} 个 key）".format(prefix, count))

    if diff["orphan_extra"]:
        ok = False
        print("\n✗ 多出来且无法解释的 key（{} 个），需要人工确认："
              .format(len(diff["orphan_extra"])))
        for key in diff["orphan_extra"][:20]:
            print("    - {} {}".format(key, list(state[key].shape)))

    if ok:
        print("\n结论：该 checkpoint 可以无损转换（只丢弃上面列出的冗余 key 与死模块）。")
    return ok


def cmd_check(args):
    net_name, cfg = resolve_config(args.exp, args.model)
    meta, state = read_checkpoint(args.checkpoint)
    model = build_model_for(net_name, cfg, parse_model_args(args.model_arg), state)
    diff = analyze(state, model)
    ok = print_report(net_name, args.checkpoint, meta, state, model, diff,
                      detect_fuse(state))
    return 0 if ok else 1


def cmd_convert(args):
    net_name, cfg = resolve_config(args.exp, args.model)
    meta, state = read_checkpoint(args.checkpoint)
    model = build_model_for(net_name, cfg, parse_model_args(args.model_arg), state)
    diff = analyze(state, model)
    ok = print_report(net_name, args.checkpoint, meta, state, model, diff,
                      detect_fuse(state))
    if not ok:
        print("\n转换中止：存在无法自动处理的结构差异。")
        return 1

    kept = {key: value for key, value in state.items() if key in set(model.state_dict())}
    try:
        model.load_state_dict(kept, strict=True)
    except RuntimeError as exc:            # 理论上不该发生，兜底
        print("\n转换失败：转出的 state_dict 无法 strict 加载：\n{}".format(exc))
        return 1
    print("\n✓ 转出的 state_dict 已用当前模型 strict 加载验证通过。")

    if args.dry_run:
        print("--dry-run：不写文件。")
        return 0

    if meta is None:
        payload = kept
    else:
        payload = dict(meta)
        payload["model_state_dict"] = kept
        if args.inference_only:
            for field in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
                payload.pop(field, None)
        else:
            print("注意：optimizer/scheduler 状态是按旧参数顺序存的，续训前请确认；"
                  "只做推理/评测可加 --inference-only 去掉它们。")
    torch.save(payload, args.out)
    print("已写入 {}".format(args.out))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, help_text in (("check", "只校验结构，不写文件"),
                            ("convert", "转换并写出新 checkpoint")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--checkpoint", required=True, help="旧代码训练出的 checkpoint")
        p.add_argument("--model", default=None, help="network.yaml 中的网络名")
        p.add_argument("--exp", default=None,
                       help="exp.yaml 中的实验名（优先级高于 --model）")
        p.add_argument("--model-arg", action="append", metavar="KEY=VALUE",
                       help="临时覆盖 model_args（如 num_classes=1296、"
                            "slowfast_config=xxx.yaml），可重复")
        if name == "convert":
            p.add_argument("--out", required=True, help="输出 checkpoint 路径")
            p.add_argument("--dry-run", action="store_true", help="只校验，不写文件")
            p.add_argument("--inference-only", action="store_true",
                           help="不保留 optimizer/scheduler/rng 状态")
            p.set_defaults(func=cmd_convert)
        else:
            p.set_defaults(func=cmd_check)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
