#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSL-Daily groundtruth 词频 / 共现统计与检索。

STM 每行形如::

    S007366_P0000_T00 1 unknown 0.0 1.79769e+308 你们 好

前 5 列(段名、通道、说话人、起始、结束)是评测文件自带的元信息,与标注内容无关,
本脚本一概不读;只有第 6 列起的空格分隔词才参与统计。整行没有词的样本(如
``S000007_P0003_T00 1 unknown 0.0 1.79769e+308``)直接跳过并计数。

统计口径(三张表,均按**语句**累计,同一个词重复出现照数):

  unigram   词 -> 出现次数
  pairs     有序同句共现:a -> {b: 次数},表示 b 出现在 a 之后的次数(不限距离)
  adjacent  紧邻对:a -> {b: 次数},表示 b 紧跟在 a 后面的次数(adjacent 是 pairs 的子集)

``pairs`` 区分先后:(你 -> 好) 与 (好 -> 你) 是两条独立记录;同一个词不与自身配对。

检索时用条件概率排序通常比原始次数更能反映关联强度:次数高的往往是「你 / 我 / 他」
这类高频词,条件概率 = 该方向上的有序共现次数 / 该词在该方向上的共现总数。

用法(仓库根目录执行)::

    # 1. 统计训练集,默认写 core/preprocess/CSL-Daily/word_cooccurrence_train.json
    python script/word_cooccurrence.py build

    # 2. 检索:它在哪些词之前 / 之后出现(默认前后各 Top 15)
    python script/word_cooccurrence.py query 你

    # 只要「紧跟在你后面」的词,按条件概率排序
    python script/word_cooccurrence.py query 你 --direction after --by adjacent --sort prob

    # 其它划分:统计文件按划分自动命名(word_cooccurrence_test.json)
    python script/word_cooccurrence.py build --stm core/preprocess/CSL-Daily/CSL-Daily-groundtruth-test.stm

    # 多个 STM 一起统计(例如 train+dev+test)
    python script/word_cooccurrence.py build \
        --stm core/preprocess/CSL-Daily/CSL-Daily-groundtruth-{train,dev,test}.stm \
        --out core/preprocess/CSL-Daily/word_cooccurrence_all.json

``query`` 找不到统计文件时,会用 ``--stm``(默认训练集)现场统计一遍,不写盘,
因此不先 build 也能直接查。

样本的多个机位(段名里的 ``_P0000`` / ``_P0004`` / ``_P0008``)是**不同的视频样本**,
默认各算一条;若希望同一句只算一次,加 ``--dedup-views``(按去掉 ``_P****`` 的段名
去重,同组只保留第一条)。注意训练集里 ``S006944_T00`` 各机位标注并不完全一致,
去重会丢掉后面几条文本身不同的标注。

输出示例见 README 或直接运行 ``query``。
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# STM 固定前 5 列是元信息(段名 通道 说话人 起始 结束),第 6 列起才是词。
STM_HEADER_FIELDS = 5

# 段名里的机位标记,如 S000000_P0004_T00 -> S000000_T00
VIEW_PATTERN = re.compile(r"_P\d+(?=_)")

DEFAULT_STM = REPO_ROOT / "core/preprocess/CSL-Daily/CSL-Daily-groundtruth-train.stm"
DEFAULT_STATS = REPO_ROOT / "core/preprocess/CSL-Daily/word_cooccurrence_train.json"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# 解析与统计
# ---------------------------------------------------------------------------

def parse_stm(path, dedup_views=False):
    """读 STM,返回 (样本列表, 跳过的无标注行数)。

    样本是 ``(段名, 词列表)``;``dedup_views`` 时按去掉机位后的段名去重,同组只留
    第一条。只有元信息、没有词的行计入 skipped,不参与任何统计。
    """
    samples = []
    skipped = 0
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) <= STM_HEADER_FIELDS:
                skipped += 1
                continue
            key = fields[0]
            if dedup_views:
                key = VIEW_PATTERN.sub("", key)
                if key in seen:
                    continue
                seen.add(key)
            samples.append((key, fields[STM_HEADER_FIELDS:]))
    return samples, skipped


def accumulate(samples):
    """把样本累计成三张统计表。

    对每条语句的每个有序位置对 (i, j),i < j,``pairs[a][b]`` 加一;j == i + 1 时
    ``adjacent[a][b]`` 再加一。同词不互配(要的是「词和其他词语」的共现)。
    """
    unigram = Counter()
    pairs = defaultdict(Counter)
    adjacent = defaultdict(Counter)
    num_tokens = 0

    for _, tokens in samples:
        num_tokens += len(tokens)
        unigram.update(tokens)
        for i, first in enumerate(tokens):
            for j in range(i + 1, len(tokens)):
                second = tokens[j]
                if first == second:
                    continue
                pairs[first][second] += 1
                if j == i + 1:
                    adjacent[first][second] += 1

    return {
        "unigram": unigram,
        "pairs": pairs,
        "adjacent": adjacent,
        "num_tokens": num_tokens,
    }


def build_stats(stm_paths, dedup_views=False):
    """统计若干 STM,返回可直接写盘的 dict。"""
    stm_paths = [Path(p) for p in stm_paths]
    samples = []
    skipped = 0
    for path in stm_paths:
        part, part_skipped = parse_stm(path, dedup_views=dedup_views)
        samples.extend(part)
        skipped += part_skipped

    stats = accumulate(samples)
    pairs = stats["pairs"]

    return {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "source_files": [relpath(p) for p in stm_paths],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dedup_views": bool(dedup_views),
            "num_samples": len(samples),
            "num_tokens": stats["num_tokens"],
            "vocab_size": len(stats["unigram"]),
            "num_pairs": sum(len(row) for row in pairs.values()),
            "num_sentences_skipped": skipped,
        },
        "unigram": dict(stats["unigram"]),
        "pairs": {k: dict(v) for k, v in pairs.items()},
        "adjacent": {k: dict(v) for k, v in stats["adjacent"].items()},
    }


def relpath(path):
    """尽量存成相对仓库根的路径,换机器后 JSON 里的来源仍可读。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------

def invert(table):
    """把 ``a -> {b: n}`` 翻成 ``b -> {a: n}``,用于查「谁出现在这个词之前」。"""
    flipped = defaultdict(dict)
    for first, row in table.items():
        for second, count in row.items():
            flipped[second][first] = count
    return flipped


def build_indexes(stats, by, needs_before):
    """按需准备四张查表:排序指标 / 对照指标,各自的正查与反查。

    ``pairs`` 与 ``adjacent`` 都只存「前 -> 后」方向,查「谁在这个词之前」要把
    表翻过来;翻表是 O(全部共现对),所以统一在这里做一次,别按词重复翻。
    """
    other = "adjacent" if by == "pairs" else "pairs"
    return {
        "after": stats[by],
        "other_after": stats[other],
        "before": invert(stats[by]) if needs_before else {},
        "other_before": invert(stats[other]) if needs_before else {},
    }


def query_word(word, stats, indexes, args):
    """检索一个词,返回结果 dict(供 --json 和打印共用)。"""
    occurrence = stats["unigram"].get(word, 0)
    if not occurrence:
        if not args.json:
            print(f"\n词「{word}」不在统计中(词表 {len(stats['unigram'])} 个词)。")
        return {"word": word, "found": False}

    result = {
        "word": word,
        "found": True,
        "occurrence": occurrence,
        "token_ratio": occurrence / max(stats["meta"]["num_tokens"], 1),
        "after": rank(indexes["after"].get(word, {}),
                      indexes["other_after"].get(word, {}), args),
        "before": rank(indexes["before"].get(word, {}),
                       indexes["other_before"].get(word, {}), args),
    }
    if not args.json:
        print_word(word, result, args)
    return result


def rank(row, other_row, args):
    """把一行 ``{词: 次数}`` 排成 ``{total, words, top:[...]}``。

    ``total`` 是这一行的**全部**共现对数,与 ``--top`` / ``--min-count`` 无关,
    是条件概率的分母;``top`` 才是要展示的条目。
    """
    total = sum(row.values())
    items = [(w, c) for w, c in row.items() if c >= args.min_count]
    if args.sort == "prob" and total:
        items.sort(key=lambda kv: (-kv[1] / total, kv[0]))
    else:
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    return {
        "total": total,
        "words": len(row),
        "top": [{"word": w, "count": c,
                 "ratio": c / total if total else 0.0,
                 "other_count": other_row.get(w, 0)} for w, c in items[:args.top]],
    }


def print_word(word, result, args):
    metric = "紧邻" if args.by == "adjacent" else "同句"
    other = "同句" if args.by == "adjacent" else "紧邻"
    print(f"\n词「{word}」: 出现 {result['occurrence']} 次 "
          f"(占总词数 {result['token_ratio']:.2%})")
    keys = {"both": ("after", "before"), "after": ("after",), "before": ("before",)}
    for key in keys[args.direction]:
        direction = "之后" if key == "after" else "之前"
        ranked = result[key]
        if not ranked["top"]:
            print(f"  「{word}」{direction}没有共现记录。")
            continue
        print(f"  「{word}」{direction}出现的词:共 {ranked['total']} 对、涉及 "
              f"{ranked['words']} 个词,下列 {len(ranked['top'])} 个按"
              f"{'条件概率' if args.sort == 'prob' else '次数'}排序")
        header = ["词", f"{metric}在{direction[-1]}", "条件概率", f"{other}在{direction[-1]}"]
        body = [[r["word"], str(r["count"]), f"{r['ratio']:.2%}", str(r["other_count"])]
                for r in ranked["top"]]
        print_table(header, body)


def print_table(header, body):
    widths = [max(width(header[i]), *(width(row[i]) for row in body))
              for i in range(len(header))]
    print("    " + "  ".join(pad(header[i], widths[i]) for i in range(len(header))))
    for row in body:
        print("    " + "  ".join(pad(row[i], widths[i], right=i > 0) for i in range(len(row))))


def width(text):
    """按终端显示宽度算(中日韩全角字符占 2 列),让中英混排的表对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, total, right=False):
    spaces = " " * max(total - width(text), 0)
    return spaces + text if right else text + spaces


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------

def load_stats(path, stm_paths, dedup_views):
    """读统计文件;文件不存在就用 STM 现场统计(不写盘)。"""
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f), path
    missing = [str(p) for p in stm_paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"统计文件 {path} 不存在,也找不到 STM: {', '.join(missing)}\n"
                         f"先运行: python {relpath(__file__)} build")
    print(f"统计文件 {relpath(path)} 不存在,改用 "
          f"{', '.join(relpath(p) for p in stm_paths)} 现场统计",
          file=sys.stderr)
    return build_stats(stm_paths, dedup_views=dedup_views), path


def default_stats_path(stm_paths):
    """按划分命名:...-groundtruth-train.stm -> word_cooccurrence_train.json。"""
    modes = []
    for path in stm_paths:
        match = re.search(r"groundtruth-(.+?)\.stm$", Path(path).name)
        modes.append(match.group(1) if match else Path(path).stem)
    return Path(stm_paths[0]).parent / f"word_cooccurrence_{'+'.join(modes)}.json"


def cmd_build(args):
    stats = build_stats(args.stm, dedup_views=args.dedup_views)
    out = Path(args.out) if args.out else default_stats_path(args.stm)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1, sort_keys=True)

    meta = stats["meta"]
    top = sorted(stats["unigram"].items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    print(f"来源 {', '.join(meta['source_files'])}")
    print(f"  样本 {meta['num_samples']} 条(跳过 {meta['num_sentences_skipped']} 条无标注行)"
          f"{' [已按机位去重]' if meta['dedup_views'] else ''}")
    print(f"  词表 {meta['vocab_size']} | 总词数 {meta['num_tokens']} "
          f"| 有序共现对 {meta['num_pairs']}")
    print("  高频词 " + ", ".join(f"{w}({c})" for w, c in top))
    print(f"已写入 {relpath(out)} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_query(args):
    stats, source = load_stats(args.stats, args.stm, args.dedup_views)
    if not args.json:
        meta = stats["meta"]
        print(f"统计文件 {relpath(source)}")
        print(f"  样本 {meta['num_samples']} 条 | 词表 {meta['vocab_size']} "
              f"| 总词数 {meta['num_tokens']} | 来源 {', '.join(meta['source_files'])}")

    indexes = build_indexes(stats, args.by, needs_before=args.direction != "after")
    results = [query_word(word, stats, indexes, args) for word in args.words]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0 if any(r["found"] for r in results) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="统计 STM 并写出 JSON")
    b.add_argument("--stm", nargs="+", type=Path, default=[DEFAULT_STM],
                   help="groundtruth STM,可给多个一起统计(默认训练集)")
    b.add_argument("--out", type=Path, default=None,
                   help="输出 JSON(默认与 STM 同目录的 word_cooccurrence_<划分>.json)")
    b.add_argument("--dedup-views", action="store_true",
                   help="同一句的多个机位只算一次(默认各算一条)")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="检索词的共现词频")
    q.add_argument("words", nargs="+", help="要检索的词,如 你 好")
    q.add_argument("--stats", type=Path, default=DEFAULT_STATS, help="统计文件")
    q.add_argument("--stm", nargs="+", type=Path, default=[DEFAULT_STM],
                   help="统计文件缺失时用它现场统计(默认训练集)")
    q.add_argument("--top", type=int, default=15, help="每个方向返回的词数(默认 15)")
    q.add_argument("--sort", choices=["count", "prob"], default="count",
                   help="排序依据:共现次数(count)或条件概率(prob)")
    q.add_argument("--by", choices=["pairs", "adjacent"], default="pairs",
                   help="排序与概率用哪张表:同句有序共现(pairs)或紧邻(adjacent)")
    q.add_argument("--min-count", type=int, default=1, help="过滤低于该共现次数的词")
    q.add_argument("--direction", choices=["both", "before", "after"], default="both",
                   help="只看某词之前(before)/之后(after)出现的词")
    q.add_argument("--dedup-views", action="store_true", help="现场统计时按机位去重")
    q.add_argument("--json", action="store_true", help="输出 JSON,便于脚本处理")
    q.set_defaults(func=cmd_query)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
