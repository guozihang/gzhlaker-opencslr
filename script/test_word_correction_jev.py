#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""word_correction_jev.py 的单元测试(纯标准库,不联网、不需要 API key)。

覆盖两件容易出错的事:

  1. 逐词词频检索与候选生成——候选只来自语料共现、排除原词、缺一侧上下文时不塌成 0、
     没有搭配记录的位置干脆不出候选。
  2. 答案组合——noul 概率过阈值才换、choice 选 none_of_these 时不硬猜一个词、
     模型漏答的位置标成「无答案」而不是当成保留。

用法:
    python3 script/test_word_correction_jev.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parent / "word_correction_jev.py"

# 一句构造的语料:「他 每天 来 很 多」和「他 每天 来 累 多」各 1 次,共 10 个词。
# 于是 来 之后 很 / 累 各 1 次(平手,按词序),多 之前也只有 很 / 累。
FAKE_STATS = {
    "meta": {"source_files": ["fake-groundtruth-train.stm"], "num_samples": 2,
             "num_tokens": 10, "vocab_size": 6},
    "unigram": {"他": 2, "每天": 2, "来": 2, "很": 1, "累": 1, "多": 2},
    "pairs": {"他": {"每天": 2}, "每天": {"来": 2}, "来": {"很": 1, "累": 1}},
    "adjacent": {
        "他": {"每天": 2},
        "每天": {"来": 2},
        "来": {"很": 1, "累": 1},
        "很": {"多": 1},
        "累": {"多": 1},
        "多": {},
    },
}

SENTENCE = ["他", "每天", "来", "累", "多"]


def load_module():
    spec = importlib.util.spec_from_file_location("word_correction_jev", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wj = load_module()


def test_parse_words_plain_and_stm():
    """普通词序列原样返回;整行 STM 丢掉前 5 列元信息。"""
    words, stripped = wj.parse_words(["他", "每天", "来"])
    assert words == ["他", "每天", "来"] and stripped is False

    words, stripped = wj.parse_words([], "S007366_P0000_T00 1 unknown 0.0 1.79769e+308 你们 好")
    assert words == ["你们", "好"] and stripped is True

    # 说话人列不是 unknown、时间列不是数字时不当成 STM,免得把正常句子截掉
    words, stripped = wj.parse_words(["他", "每天", "来", "累", "多", "了"])
    assert words == ["他", "每天", "来", "累", "多", "了"] and stripped is False


def test_invert_flips_direction():
    flipped = wj.invert({"a": {"b": 2, "c": 1}})
    assert flipped["b"] == {"a": 2}, flipped
    assert flipped["c"] == {"a": 1}, flipped


def test_top_neighbors_sorted_with_probability():
    top = wj.top_neighbors({"a": 1, "b": 3, "c": 3}, top=2)
    assert [t["word"] for t in top] == ["b", "c"], top      # 次数相同按词序
    assert top[0]["prob"] == 3 / 7 and top[0]["count"] == 3, top
    assert wj.top_neighbors({}, top=2) == []


def test_candidates_exclude_original_and_merge_two_sides():
    """下标 3 的原词是 累:左邻 来 之后、右邻 多 之前都只有 很 可用。

    原词不进候选,但算条件概率时分母仍是整行(原词那几次也是真实证据);
    两侧各半权,合成 0.5 * 1/2 + 0.5 * 1/2 = 0.5。
    """
    lookup = wj.build_lookup(FAKE_STATS)
    candidates = wj.candidates_at(lookup, prev="来", nxt="多", original="累", top_k=6)
    assert [c["word"] for c in candidates] == ["很"], candidates
    assert candidates[0]["after_left_neighbor"] == {"count": 1, "prob": 0.5}, candidates
    assert candidates[0]["before_right_neighbor"] == {"count": 1, "prob": 0.5}, candidates
    assert abs(candidates[0]["score"] - 0.5) < 1e-9, candidates


def test_candidates_missing_side_does_not_halve():
    """只有一侧有搭配记录时,权重全给这一侧,不因缺一侧而减半。"""
    lookup = wj.build_lookup(FAKE_STATS)
    # 每天 之后只见过 来(2 次);原词给 他,才轮得到 来 进候选
    candidates = wj.candidates_at(lookup, prev="每天", nxt=None, original="他", top_k=6)
    assert [c["word"] for c in candidates] == ["来"], candidates
    assert candidates[0]["after_left_neighbor"] == {"count": 2, "prob": 1.0}, candidates
    assert candidates[0]["before_right_neighbor"] is None, candidates
    assert abs(candidates[0]["score"] - 1.0) < 1e-9, candidates


def test_candidates_empty_when_original_is_the_only_option():
    """下标 2 的原词是 来:左右两侧的证据都指向它自己,排除后没有候选。"""
    lookup = wj.build_lookup(FAKE_STATS)
    assert wj.candidates_at(lookup, prev="每天", nxt="累", original="来", top_k=6) == []


def test_analyse_reports_frequency_for_every_word():
    reports, _, _ = wj.analyse_sample(SENTENCE, FAKE_STATS)
    assert [r["word"] for r in reports] == SENTENCE
    assert [r["index"] for r in reports] == [0, 1, 2, 3, 4]

    lei = reports[3]
    assert lei["corpus_count"] == 1, lei
    assert abs(lei["corpus_share"] - 1 / 10) < 1e-9, lei
    assert lei["left_neighbor"] == "来" and lei["right_neighbor"] == "多", lei
    # 「累」自己的常见搭配:常见后词 多(1 次)
    assert lei["word_usual_followers"] == [{"word": "多", "count": 1, "prob": 1.0}], lei
    assert [c["word"] for c in lei["candidates"]] == ["很"], lei


def test_build_questions_skips_choice_without_candidates():
    reports, state, questions = wj.analyse_sample(SENTENCE, FAKE_STATS)

    # 5 个位置都要问「要不要换」;但这份小语料里只有下标 3 能给出候选词,所以只有一个 choice
    assert set(questions) == {"replace_0", "replace_1", "replace_2", "replace_3", "replace_4",
                              "replacement_3"}, sorted(questions)
    assert "replacement_2" not in questions, "下标 2 没有候选词,不该发 choice"

    assert questions["replace_3"]["type"] == "noul"
    assert set(questions["replace_3"]["criteria"]) == {"true", "false"}
    assert questions["replace_3"]["instructions"]["local_context"] == "他 每天 来 ___ 多"

    criteria = questions["replacement_3"]["criteria"]
    assert set(criteria) == {"很", wj.NONE_OPTION}, criteria
    assert "来" in criteria["很"], criteria          # 说明里带上了语料依据

    assert state["sentence"] == "他 每天 来 累 多"
    assert state["positions"][3]["word"] == "累"
    assert "corpus_share" in state["positions"][0]


def test_state_is_json_serializable():
    """state / questions 要能直接 json.dumps(--dry-run 就是把它打出来)。"""
    _, state, questions = wj.analyse_sample(SENTENCE, FAKE_STATS)
    text = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
    assert "很" in text, "中文应原样输出"


def test_read_answers_flattens_sdk_objects():
    response = SimpleNamespace(answers={
        "replace_3": SimpleNamespace(type="noul", noul=0.97),
        "replacement_3": SimpleNamespace(type="choice", choice="很", confidence=0.62,
                                         probabilities={"很": 0.62, wj.NONE_OPTION: 0.38}),
    })
    answers = wj.read_answers(response)
    assert answers["replace_3"]["noul"] == 0.97
    assert answers["replacement_3"]["choice"] == "很"
    assert answers["replacement_3"]["probabilities"][wj.NONE_OPTION] == 0.38


def test_decide_threshold_and_none_option():
    report = {"index": 3, "word": "累"}

    # 过阈值且选了具体候选词 -> 替换
    d = wj.decide(report, {"replace_3": {"noul": 0.97},
                           "replacement_3": {"choice": "很", "confidence": 0.62}}, threshold=0.5)
    assert d["verdict"] == "replace" and d["replacement"] == "很", d

    # 没过阈值 -> 保留
    d = wj.decide(report, {"replace_3": {"noul": 0.30},
                           "replacement_3": {"choice": "很"}}, threshold=0.5)
    assert d["verdict"] == "keep" and d["replacement"] is None, d

    # 刚好等于阈值算替换(判据是 < threshold 才保留)
    d = wj.decide(report, {"replace_3": {"noul": 0.5},
                           "replacement_3": {"choice": "很"}}, threshold=0.5)
    assert d["verdict"] == "replace", d

    # 该换但候选都不合适 -> 不硬猜一个词
    d = wj.decide(report, {"replace_3": {"noul": 0.97},
                           "replacement_3": {"choice": wj.NONE_OPTION}}, threshold=0.5)
    assert d["verdict"] == "replace_no_candidate" and d["replacement"] is None, d

    # 该换但压根没发 choice(没候选) -> 同上
    d = wj.decide(report, {"replace_3": {"noul": 0.97}}, threshold=0.5)
    assert d["verdict"] == "replace_no_candidate", d

    # 模型没给答案 -> 无答案,不能当成保留
    d = wj.decide(report, {}, threshold=0.5)
    assert d["verdict"] == "unknown" and d["replace_prob"] is None, d


def test_apply_decisions_builds_corrected_sentence():
    decisions = [
        {"index": 0, "word": "他", "verdict": "keep", "replace_prob": 0.02,
         "replacement": None, "replacement_confidence": None, "replacement_probabilities": {}},
        {"index": 3, "word": "累", "verdict": "replace", "replace_prob": 0.97,
         "replacement": "很", "replacement_confidence": 0.62, "replacement_probabilities": {}},
        {"index": 4, "word": "多", "verdict": "replace_no_candidate", "replace_prob": 0.88,
         "replacement": None, "replacement_confidence": None, "replacement_probabilities": {}},
    ]
    corrected, changes = wj.apply_decisions(SENTENCE, decisions)

    assert corrected == ["他", "每天", "来", "很", "多"], corrected
    assert SENTENCE == ["他", "每天", "来", "累", "多"], "原句列表不能被就地改掉"
    assert changes == [
        {"index": 3, "from": "累", "to": "很", "replace_prob": 0.97},
        {"index": 4, "from": "多", "to": None, "replace_prob": 0.88},
    ], changes


def test_apply_decisions_no_changes():
    decisions = [{"index": 0, "word": "他", "verdict": "keep", "replace_prob": 0.01,
                  "replacement": None, "replacement_confidence": None,
                  "replacement_probabilities": {}}]
    corrected, changes = wj.apply_decisions(["他"], decisions)
    assert corrected == ["他"] and changes == []


def test_load_stats_rejects_foreign_json():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "not_stats.json"
        path.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        try:
            wj.load_stats(path)
        except SystemExit as e:
            assert "unigram" in str(e), e
        else:
            raise AssertionError("缺表的 JSON 应该报错退出")


def test_resolve_stats_path_explicit():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stats.json"
        path.write_text(json.dumps(FAKE_STATS, ensure_ascii=False), encoding="utf-8")
        assert wj.resolve_stats_path(path) == path


def test_main_end_to_end_with_fake_jev():
    """整条链路跑一遍(用假 response 顶替 API):打印、判定、纠正后的句子都对。"""
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "word_cooccurrence_fake.json"
        stats_path.write_text(json.dumps(FAKE_STATS, ensure_ascii=False), encoding="utf-8")

        fake_answers = {f"replace_{i}": SimpleNamespace(type="noul", noul=0.02) for i in range(5)}
        fake_answers["replace_3"] = SimpleNamespace(type="noul", noul=0.97)
        fake_answers["replacement_3"] = SimpleNamespace(
            type="choice", choice="很", confidence=0.62,
            probabilities={"很": 0.62, wj.NONE_OPTION: 0.38})
        fake_response = SimpleNamespace(
            model="jev-fake", usage=SimpleNamespace(input_tokens=321, output_tokens=48),
            answers=fake_answers)

        original_call, original_argv = wj.call_jev, sys.argv
        wj.call_jev = lambda state, questions, model, api_key: fake_response
        sys.argv = ["word_correction_jev.py", "他", "每天", "来", "累", "多",
                    "--stats", str(stats_path), "--api-key", "fake-key"]
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = wj.main()
        finally:
            wj.call_jev, sys.argv = original_call, original_argv

    out = buffer.getvalue()
    assert code == 0, code
    assert "原句    他 每天 来 累 多" in out, out
    assert "纠正后  他 每天 来 很 多" in out, out
    assert "「累」-> 「很」" in out, out
    assert "token: 输入 321 / 输出 48" in out, out


def test_main_dry_run_makes_no_api_call():
    """--dry-run 只打印请求体,不碰密钥、不发请求。"""
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "word_cooccurrence_fake.json"
        stats_path.write_text(json.dumps(FAKE_STATS, ensure_ascii=False), encoding="utf-8")

        def explode(*_args, **_kwargs):
            raise AssertionError("--dry-run 不应该调用 jev")

        original_call, original_argv = wj.call_jev, sys.argv
        original_key = wj.resolve_api_key
        wj.call_jev = explode
        wj.resolve_api_key = explode
        sys.argv = ["word_correction_jev.py", "他", "每天", "来", "累", "多",
                    "--stats", str(stats_path), "--dry-run"]
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = wj.main()
        finally:
            wj.call_jev, wj.resolve_api_key, sys.argv = original_call, original_key, original_argv

    body = json.loads(buffer.getvalue())
    assert code == 0, code
    assert body["model"] and "state" in body and "questions" in body, body
    assert body["state"]["sentence"] == "他 每天 来 累 多"
    assert set(body["questions"]) == {"replace_0", "replace_1", "replace_2", "replace_3",
                                      "replace_4", "replacement_3"}, body["questions"]


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
