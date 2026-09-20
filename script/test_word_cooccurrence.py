#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""word_cooccurrence.py 的单元测试(纯标准库,不需要数据集与 GPU)。

用一段构造的 STM 文本验证解析与统计口径:跳过没有标注的行、区分 (a -> b) 与
(b -> a)、紧邻表是共现表的子集、同词不自配、去重后同句只算一次。

用法:
    python3 script/test_word_cooccurrence.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parent / "word_cooccurrence.py"

# 前 3 行是 S000001 / S000002 的样本,最后一行只有元信息、没有词
FAKE_STM = """\
S000001_P0000_T00 1 unknown 0.0 1.79769e+308 你 好
S000001_P0004_T00 1 unknown 0.0 1.79769e+308 你 好
S000002_P0000_T00 1 unknown 0.0 1.79769e+308 好 你 好
S000003_P0000_T00 1 unknown 0.0 1.79769e+308
"""


def load_module():
    spec = importlib.util.spec_from_file_location("word_cooccurrence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wc = load_module()


def write_fake_stm(tmpdir, text=FAKE_STM):
    path = Path(tmpdir) / "FAKE-groundtruth-train.stm"
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_skips_header_only_lines():
    """只有 5 列元信息的行要跳过,并计入 num_sentences_skipped。"""
    with tempfile.TemporaryDirectory() as tmp:
        samples, skipped = wc.parse_stm(write_fake_stm(tmp))
    assert skipped == 1, skipped
    assert [key for key, _ in samples] == \
        ["S000001_P0000_T00", "S000001_P0004_T00", "S000002_P0000_T00"]
    assert samples[2][1] == ["好", "你", "好"], samples[2]


def test_counts_pair_ordering_and_adjacency():
    """手算期望:你->好 3 次(全部紧邻),好->你 1 次;同词(好 好)不自配。"""
    with tempfile.TemporaryDirectory() as tmp:
        stats = wc.build_stats([write_fake_stm(tmp)])

    assert stats["unigram"] == {"你": 3, "好": 4}, stats["unigram"]
    # 你 好 / 你 好 / 好 [你 好] —— 三处 你 都在 好 之前
    assert stats["pairs"]["你"] == {"好": 3}, stats["pairs"]
    # 只有第二句的 好 出现在 你 之前
    assert stats["pairs"]["好"] == {"你": 1}, stats["pairs"]
    assert stats["adjacent"]["你"] == {"好": 3}, stats["adjacent"]
    assert stats["adjacent"]["好"] == {"你": 1}, stats["adjacent"]

    meta = stats["meta"]
    assert meta["num_samples"] == 3, meta
    assert meta["num_sentences_skipped"] == 1, meta
    assert meta["num_tokens"] == 7, meta
    assert meta["vocab_size"] == 2, meta


def test_dedup_views_keeps_one_per_sentence():
    """--dedup-views 后 S000001 的第二个机位不再重复计数。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_fake_stm(tmp)
        plain = wc.build_stats([path])
        deduped = wc.build_stats([path], dedup_views=True)

    assert deduped["meta"]["num_samples"] == 2, deduped["meta"]
    assert deduped["unigram"] == {"你": 2, "好": 3}, deduped["unigram"]
    assert deduped["pairs"]["你"] == {"好": 2}, deduped["pairs"]
    assert plain["unigram"]["你"] == 3, plain["unigram"]   # 默认不去重


def test_adjacent_is_subset_of_pairs():
    """任意距离的共现包含紧邻,所以每对 adjacent <= pairs。"""
    with tempfile.TemporaryDirectory() as tmp:
        stats = wc.build_stats([write_fake_stm(tmp)])
    for first, row in stats["adjacent"].items():
        for second, count in row.items():
            assert count <= stats["pairs"][first][second], (first, second)


def test_query_direction_is_respected():
    """同一对词两个方向是两条独立记录:你->好 3 次,好->你 1 次。"""
    with tempfile.TemporaryDirectory() as tmp:
        stats = wc.build_stats([write_fake_stm(tmp)])
    indexes = wc.build_indexes(stats, by="pairs", needs_before=True)
    args = SimpleNamespace(top=5, sort="count", min_count=1, by="pairs",
                           direction="both", json=True)

    # 查 你:好 在后面 3 次、在前面 1 次,两个方向各算各的
    ni = wc.query_word("你", stats, indexes, args)
    assert ni["found"] and ni["occurrence"] == 3
    assert [r["word"] for r in ni["after"]["top"]] == ["好"], ni["after"]
    assert ni["after"]["total"] == 3 and ni["after"]["words"] == 1, ni["after"]
    assert ni["after"]["top"][0]["count"] == 3, ni["after"]
    assert ni["after"]["top"][0]["other_count"] == 3, ni["after"]   # 3 次全部紧邻
    assert [r["word"] for r in ni["before"]["top"]] == ["好"], ni["before"]
    assert ni["before"]["top"][0]["count"] == 1, ni["before"]

    # 查 好:方向刚好反过来(反查表确实翻过方向)
    hao = wc.query_word("好", stats, indexes, args)
    assert [r["word"] for r in hao["after"]["top"]] == ["你"], hao["after"]
    assert hao["after"]["top"][0]["count"] == 1, hao["after"]
    assert [r["word"] for r in hao["before"]["top"]] == ["你"], hao["before"]
    assert hao["before"]["top"][0]["count"] == 3, hao["before"]

    missing = wc.query_word("没这个词", stats, indexes, args)
    assert missing == {"word": "没这个词", "found": False}, missing


def test_min_count_and_top_filter():
    with tempfile.TemporaryDirectory() as tmp:
        stats = wc.build_stats([write_fake_stm(tmp)])
    indexes = wc.build_indexes(stats, by="pairs", needs_before=False)
    args = SimpleNamespace(top=1, sort="prob", min_count=4, by="pairs",
                           direction="after", json=True)
    # 你->好 只有 3 次,低于 min_count=4,应被过滤掉,但 total 仍报全部 3 对
    result = wc.query_word("你", stats, indexes, args)["after"]
    assert result["top"] == [], result
    assert result["total"] == 3, result


def test_default_stats_path_follows_split():
    path = wc.default_stats_path([Path("core/preprocess/CSL-Daily/CSL-Daily-groundtruth-dev.stm")])
    assert path.name == "word_cooccurrence_dev.json", path
    assert path.parent.name == "CSL-Daily", path


def test_json_roundtrip():
    """写出的 JSON 能被 query 侧直接读回(键都是字符串,中文不转义)。"""
    with tempfile.TemporaryDirectory() as tmp:
        stats = wc.build_stats([write_fake_stm(tmp)])
        out = Path(tmp) / "stats.json"
        out.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
        text = out.read_text(encoding="utf-8")
        loaded = json.loads(text)
    assert loaded["unigram"]["你"] == 3
    assert "你" in text, "中文应原样写入,不转成 \\uXXXX"


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:                      # 打印失败详情,跑完所有用例
            failed += 1
            print(f"[FAIL] {test.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"[OK]   {test.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
