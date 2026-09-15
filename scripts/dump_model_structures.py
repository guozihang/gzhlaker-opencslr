#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型结构对比脚本（重构验收用）。

重构模型代码时唯一不能变的东西是 state_dict：参数名、形状、权重共享关系必须逐一
相同，否则已有的 checkpoint 全部作废。本脚本把每个网络的这些信息导成 JSON，用来
对比重构前后两侧的代码。

用法（仓库根目录执行）:

    # 在旧代码那一侧（重构前的 commit）
    python scripts/dump_model_structures.py dump --out /tmp/before.json

    # 在新代码这一侧（重构后）
    python scripts/dump_model_structures.py dump --out /tmp/after.json

    # 对比；退出码 0 = 完全一致，1 = 有差异（逐条列出）
    python scripts/dump_model_structures.py compare /tmp/before.json /tmp/after.json

若只关心某个实验（实验节会覆盖网络节的同名键，与 ConfigManager 一致）:

    python scripts/dump_model_structures.py dump --exp baseline --out /tmp/before.json

说明:
  - 只做 CPU 构建，不需要 GPU。模型按网络名从注册表取构建函数，因此本脚本同时
    验证了「每个注册名都能构建出模型」。
  - gloss_dict 用合成的假词表（大小 num_classes-1）；它只喂给解码器，不产生参数，
    不影响 dump 结果。
  - 若某个网络在构造期会下载/加载预训练权重（如 SlowFast），需要权重已就绪或网络
    可达，否则该节会记成构建失败——对比时两侧报同样的失败即视为一致。
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = REPO_ROOT / "core"
sys.path.insert(0, str(CORE))

import yaml  # noqa: E402


def load_sections(exp_name=None):
    """返回 {网络名: 配置}。

    exp_name 为 None 时取 network.yaml 的全部节；否则按 ConfigManager.load_experiment
    的规则合并该实验（注意是**浅**合并：实验节的 model_args 会整体覆盖网络节的）。
    """
    with open(CORE / "configs" / "network.yaml", encoding="utf-8") as f:
        networks = yaml.load(f, Loader=yaml.FullLoader) or {}
    if exp_name is None:
        return {k: v for k, v in networks.items() if not k.startswith("_")}

    with open(CORE / "configs" / "exp.yaml", encoding="utf-8") as f:
        exps = yaml.load(f, Loader=yaml.FullLoader) or {}
    if exp_name not in exps:
        raise SystemExit(f"exp.yaml 中没有实验 {exp_name!r}；可用: {sorted(exps)}")
    exp = exps[exp_name]
    net_name = exp.get("network")
    if not net_name:
        raise SystemExit(f"实验 {exp_name!r} 未指定 network")
    merged = dict(networks[net_name])
    merged.update(exp)              # 与 ConfigManager.load_experiment 一致
    merged.pop("network", None)
    return {net_name: merged}


def structure_of(net_name, cfg):
    """构建模型并导出其参数结构。构建失败时返回错误信息而不是抛异常。"""
    import torch  # noqa: F401
    from models import get_model_builder

    model_args = dict(cfg.get("model_args") or {})
    num_classes = int(model_args["num_classes"])
    gloss_dict = {i: (f"gloss{i}",) for i in range(num_classes - 1)}

    builder = get_model_builder(cfg.get("model", net_name))
    model = builder(model_args, gloss_dict=gloss_dict,
                    loss_weights=cfg.get("loss_weights") or {})

    # named_parameters() 默认对共享权重去重，只留其中一个名字
    params = {}
    for pname, p in model.named_parameters():
        params[pname] = {"shape": list(p.shape), "requires_grad": bool(p.requires_grad)}

    sd = model.state_dict()
    buffer_names = set(n for n, _ in model.named_buffers())
    # state_dict() 不做去重：共享权重会以多个名字出现，据此还原共享关系。
    # 只用 state_dict()/named_buffers() 这些老 API，避免依赖 torch>=1.9 的
    # named_parameters(remove_duplicate=False)。
    by_ptr = {}
    for pname in sorted(set(sd) - buffer_names):
        tensor = sd[pname]
        if tensor.numel() == 0:
            continue                    # 空张量的 data_ptr 不可靠
        by_ptr.setdefault(tensor.data_ptr(), []).append(pname)
    shared = sorted(sorted(v) for v in by_ptr.values() if len(v) > 1)

    return {
        "parameters": params,
        "shared_groups": shared,
        "buffer_names": sorted(buffer_names),
    }


def cmd_dump(args):
    sections = load_sections(args.exp)
    out = {}
    for net_name in sorted(sections):
        try:
            out[net_name] = structure_of(net_name, sections[net_name])
            print(f"  {net_name:12s} {len(out[net_name]['parameters']):4d} 个参数, "
                  f"{len(out[net_name]['shared_groups'])} 组共享")
        except Exception as e:                      # 记录失败，供两侧对比
            out[net_name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {net_name:12s} 构建失败: {type(e).__name__}: {e}")
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"已写入 {args.out}")


def cmd_compare(args):
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    problems = []

    for net in sorted(set(before) | set(after)):
        if net not in before:
            problems.append(f"[{net}] 只出现在 after（新增网络）")
            continue
        if net not in after:
            problems.append(f"[{net}] 只出现在 before（网络消失）")
            continue
        b, a = before[net], after[net]
        if "error" in b or "error" in a:
            if b.get("error") != a.get("error"):
                problems.append(f"[{net}] 构建失败情况不一致:\n    before: {b.get('error')}\n    after:  {a.get('error')}")
            continue

        bp, ap = b["parameters"], a["parameters"]
        for k in sorted(set(bp) | set(ap)):
            if k not in bp:
                problems.append(f"[{net}] 新增参数 {k} {ap[k]['shape']}")
            elif k not in ap:
                problems.append(f"[{net}] 丢失参数 {k} {bp[k]['shape']}")
            elif bp[k]["shape"] != ap[k]["shape"]:
                problems.append(f"[{net}] {k} 形状变化 {bp[k]['shape']} -> {ap[k]['shape']}")
            elif bp[k]["requires_grad"] != ap[k]["requires_grad"]:
                problems.append(f"[{net}] {k} requires_grad 变化 "
                                f"{bp[k]['requires_grad']} -> {ap[k]['requires_grad']}")

        if b["shared_groups"] != a["shared_groups"]:
            problems.append(f"[{net}] 权重共享关系变化:\n"
                            f"    before: {b['shared_groups']}\n"
                            f"    after:  {a['shared_groups']}")
        if b["buffer_names"] != a["buffer_names"]:
            only_b = sorted(set(b["buffer_names"]) - set(a["buffer_names"]))
            only_a = sorted(set(a["buffer_names"]) - set(b["buffer_names"]))
            if only_b:
                problems.append(f"[{net}] buffer 消失: {only_b}")
            if only_a:
                problems.append(f"[{net}] buffer 新增: {only_a}")

    if problems:
        print(f"发现 {len(problems)} 处差异：\n")
        for p in problems:
            print("  ✗ " + p)
        return 1

    total = sum(len(v.get("parameters", {})) for v in after.values())
    print(f"✓ {len(after)} 个网络、{total} 个参数的名称/形状/共享关系/requires_grad 完全一致")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="构建各网络的模型并导出结构 JSON")
    d.add_argument("--out", required=True, help="输出 JSON 路径")
    d.add_argument("--exp", default=None,
                   help="只导出该实验引用的网络（实验节覆盖网络节的同名键）")
    d.set_defaults(func=cmd_dump)

    c = sub.add_parser("compare", help="对比两份结构 JSON")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
