#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 TypeSafe 的 jev 模型逐词纠正 CSL-Daily gloss 样本。

流程分两步,职责分明:

  1. **检索词频(纯代码)**:对样本里的每个词,从 ``word_cooccurrence.py build``
     产出的统计 JSON 里查它的词频、它常与哪些词相邻,并给出**候选替换词**——
     候选只来自语料共现:左邻词后面常接的词 ∪ 右邻词前面常接的词,按条件概率
     合成打分排序。这一步是确定性的,不看模型。
  2. **逐词判断(jev)**:把整句话 + 每个位置的词频证据作为 state 一次性发给
     jev,每个位置问两个问题,并行返回:

       replace_<i>      noul    这个位置的词是否**应当**被替换(1=应替换)
       replacement_<i>  choice  如果替换,换成哪个候选词(候选都不合适选 none_of_these)

     最后代码按 ``--threshold`` 组合两份答案,输出纠正后的句子。

两个问题在**同一次请求**里问(Speculative fan-out),所以一次调用就能拿到全句
的判断;哪个位置没有候选词就不发对应的 choice。

用法(仓库根目录执行)::

    # 默认读 core/preprocess/CSL-Daily/word_cooccurrence_train.json
    python script/word_correction_jev.py 他 每天 来 累 多
    python script/word_correction_jev.py --sentence "他 每天 来 累 多"

    # 看请求体,不调 API(可直接粘到 Playground 或 curl)
    python script/word_correction_jev.py 他 每天 来 累 多 --dry-run

    # 换统计文件 / 调阈值 / 多给候选
    python script/word_correction_jev.py 他 每天 来 累 多 \
        --stats ~/Downloads/word_cooccurrence_CSL-Daily_train.json \
        --threshold 0.6 --top-k 8 --json

输入既可以直接给词,也可以整行粘贴 STM(前 5 列元信息会自动丢掉)::

    python script/word_correction_jev.py "S007366_P0000_T00 1 unknown 0.0 1.79769e+308 你们 好"

Google Colab 用法::

    !pip install typesafe-sdk
    # 再把 word_cooccurrence_CSL-Daily_train.json 和本脚本传上去(左侧「文件」面板)
    # 密钥:左侧 🔑 图标里加一个 TYPESAFE_API_KEY,或在 cell 里 %env TYPESAFE_API_KEY=...
    %run word_correction_jev.py 他 每天 来 累 多

本脚本刻意做成**单文件**:只依赖标准库 + ``typesafe-sdk``,不 import 同目录的
``word_cooccurrence.py``,这样往 Colab 上传一个文件就能跑。表头对齐那三个小函数
与 ``word_cooccurrence.py`` 是重复的,改动时请两边一起改。

环境变量:
    TYPESAFE_API_KEY       必填(也可用 Colab secret 或 --api-key)
    TYPESAFE_DEFAULT_MODEL 可选,默认 jev-latest
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# STM 固定前 5 列是元信息(段名 通道 说话人 起始 结束),第 6 列起才是词。
STM_HEADER_FIELDS = 5
STM_SEGMENT_PATTERN = re.compile(r"^\S+_\S+_\S+$")

DEFAULT_STATS = REPO_ROOT / "core/preprocess/CSL-Daily/word_cooccurrence_train.json"
DEFAULT_MODEL = "jev-latest"
API_KEY_ENV = "TYPESAFE_API_KEY"

# choice 里「候选都不合适」的哨兵选项。它不是词,不会和统计词表里的词重名。
NONE_OPTION = "none_of_these"
NONE_DESCRIPTION = "以上候选都不合适:这个位置应该换成列表之外的其它词,或需要人工确认"

DEFAULT_THRESHOLD = 0.5
DEFAULT_TOP_K = 6
DEFAULT_CONTEXT_TOP = 4


# ---------------------------------------------------------------------------
# 第一步:词频检索(纯代码,确定性)
# ---------------------------------------------------------------------------

def parse_words(positional, sentence=None):
    """把命令行输入拼成词序列,并识别整行 STM(丢掉前 5 列元信息)。

    返回 ``(词列表, 是否丢掉过 STM 表头)``。
    """
    text = sentence if sentence else " ".join(positional)
    tokens = text.split()
    if len(tokens) <= STM_HEADER_FIELDS:
        return tokens, False
    if not STM_SEGMENT_PATTERN.match(tokens[0]):
        return tokens, False
    try:                                   # STM 的起始/结束时间是浮点数,说话人列是 unknown
        float(tokens[3])
        float(tokens[4])
    except ValueError:
        return tokens, False
    return tokens[STM_HEADER_FIELDS:], True


def load_stats(path):
    """读 word_cooccurrence.py 产出的统计 JSON,缺表就报错退出。"""
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    for key in ("unigram", "pairs", "adjacent"):
        if key not in stats:
            raise SystemExit(f"{path} 不像是 word_cooccurrence.py build 出来的统计文件"
                             f"(缺少 {key!r});先运行 python script/word_cooccurrence.py build")
    stats.setdefault("meta", {})
    return stats


def invert(table):
    """把 ``a -> {b: n}`` 翻成 ``b -> {a: n}``,用来查「谁紧邻在 b 之前」。"""
    flipped = defaultdict(dict)
    for first, row in table.items():
        for second, count in row.items():
            flipped[second][first] = count
    return flipped


def build_lookup(stats):
    """整理出逐词检索用的三张表。紧邻(adjacent)是这里的主角:判断的是某个位置
    该放哪个词,距离为 1 的搭配比同句任意距离的共现更有说服力。"""
    return {
        "unigram": stats["unigram"],
        "after": stats["adjacent"],           # a -> {b: n},b 紧跟在 a 之后
        "before": invert(stats["adjacent"]),  # b -> {a: n},a 紧邻在 b 之前
        "num_tokens": stats["meta"].get("num_tokens") or sum(stats["unigram"].values()),
    }


def top_neighbors(row, top):
    """把一行 ``{词: 次数}`` 排成按次数递减的 Top N,并给出条件概率。"""
    total = sum(row.values())
    items = sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"word": w, "count": c, "prob": c / total if total else 0.0} for w, c in items]


def candidates_at(lookup, prev, nxt, original, top_k):
    """位置 i 的候选替换词:左邻词之后常接的词 ∪ 右邻词之前常接的词。

    两侧各按条件概率给一半权重(缺一侧就全给另一侧),再按合成分排序取前 top_k。
    原词本身不进候选——它已经在 state 里,而且这里要的是「换成什么」。
    """
    sides = []
    if prev is not None and lookup["after"].get(prev):
        sides.append(("after_left_neighbor", lookup["after"][prev]))
    if nxt is not None and lookup["before"].get(nxt):
        sides.append(("before_right_neighbor", lookup["before"][nxt]))
    if not sides:
        return []

    weight = 1.0 / len(sides)
    scores = defaultdict(float)
    evidence = defaultdict(dict)
    for name, row in sides:
        total = sum(row.values())
        for word, count in row.items():
            if word == original:
                continue
            scores[word] += weight * count / total
            evidence[word][name] = {"count": count, "prob": count / total}

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [{"word": word, "score": score,
             "after_left_neighbor": evidence[word].get("after_left_neighbor"),
             "before_right_neighbor": evidence[word].get("before_right_neighbor")}
            for word, score in ranked]


def analyse(words, lookup, top_k=DEFAULT_TOP_K, context_top=DEFAULT_CONTEXT_TOP):
    """对样本里的每个词检索词频,返回逐位置的证据列表。"""
    reports = []
    for i, word in enumerate(words):
        prev = words[i - 1] if i > 0 else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        count = lookup["unigram"].get(word, 0)
        reports.append({
            "index": i,
            "word": word,
            "corpus_count": count,
            "corpus_share": count / max(lookup["num_tokens"], 1),
            "left_neighbor": prev,
            "right_neighbor": nxt,
            "word_usual_preceders": top_neighbors(lookup["before"].get(word, {}), context_top),
            "word_usual_followers": top_neighbors(lookup["after"].get(word, {}), context_top),
            "after_left_neighbor": top_neighbors(lookup["after"].get(prev, {}), context_top),
            "before_right_neighbor": top_neighbors(lookup["before"].get(nxt, {}), context_top),
            "candidates": candidates_at(lookup, prev, nxt, word, top_k),
        })
    return reports


# ---------------------------------------------------------------------------
# 第二步:组装 state / questions,交给 jev
# ---------------------------------------------------------------------------

def sentence_with_gap(words, index):
    """把第 index 个词换成 ``___``,给模型一个「这个位置缺词」的视角。"""
    return " ".join("___" if i == index else w for i, w in enumerate(words))


def build_state(words, reports, stats):
    """请求的 state:整句话 + 每个位置的词频证据,问题里用 ``positions[i]`` 指过来。"""
    meta = stats["meta"]
    return {
        "task": "逐词检查一句 CSL-Daily 手语标注(gloss)用词是否正确,并给出应当替换成的词",
        "corpus": {
            "name": meta.get("source_files") or "CSL-Daily groundtruth 词频共现统计",
            "num_samples": meta.get("num_samples"),
            "num_tokens": meta.get("num_tokens"),
            "vocab_size": meta.get("vocab_size"),
            "note": "`after_left_neighbor` 是语料里紧跟在左邻词之后出现的词;"
                    "`before_right_neighbor` 是语料里紧邻在右邻词之前出现的词;"
                    "`prob` 是条件概率,`count` 是原始次数。",
        },
        "sentence": " ".join(words),
        "positions": [
            {
                "index": r["index"],
                "word": r["word"],
                "corpus_count": r["corpus_count"],
                "corpus_share": r["corpus_share"],
                "left_neighbor": r["left_neighbor"],
                "right_neighbor": r["right_neighbor"],
                "word_usual_preceders": r["word_usual_preceders"],
                "word_usual_followers": r["word_usual_followers"],
                "after_left_neighbor": r["after_left_neighbor"],
                "before_right_neighbor": r["before_right_neighbor"],
                "replacement_candidates": [
                    {"word": c["word"], "score": c["score"],
                     "after_left_neighbor": c["after_left_neighbor"],
                     "before_right_neighbor": c["before_right_neighbor"]}
                    for c in r["candidates"]
                ],
            }
            for r in reports
        ],
    }


def describe_candidate(candidate, report):
    """候选词在 choice 里的说明:说清它为什么被选中,全部来自语料统计。"""
    parts = []
    left = candidate["after_left_neighbor"]
    if left:
        parts.append(f"语料里紧跟「{report['left_neighbor']}」之后出现 {left['count']} 次"
                     f"(条件概率 {left['prob']:.2%})")
    right = candidate["before_right_neighbor"]
    if right:
        parts.append(f"语料里紧邻「{report['right_neighbor']}」之前出现 {right['count']} 次"
                     f"(条件概率 {right['prob']:.2%})")
    if not parts:
        parts.append("语料里与左右邻词的搭配记录很少")
    return ";".join(parts)


def build_questions(reports, words):
    """每个位置两个问题:要不要换(noul)+ 换成哪个(choice),一次请求一起问。

    用普通的 dict 而不是 SDK 的 ``Noul`` / ``Choice`` 对象:dict 就是 API 的请求体
    形态,SDK 与 --dry-run 的 JSON 都能直接吃。
    """
    questions = {}
    for report in reports:
        i = report["index"]
        questions[f"replace_{i}"] = {
            "type": "noul",
            "instructions": {
                "question": f"`sentence` 里第 {i + 1} 个词(下标 {i},也就是 `positions[{i}].word`)"
                            f"用在这里是否应当被替换掉?",
                "local_context": sentence_with_gap(words, i),
                "guidance": f"结合 `positions[{i}]` 里的语料统计判断:如果这个词与左右邻词的搭配"
                            f"在语料里几乎不出现,或者它在这句话里语义不通、明显是听错/记错的词,"
                            f"就应当替换;如果它是这句话里合理自然的词,就应当保留。",
            },
            "criteria": {
                "true": "这个位置用词不当,应当换成别的词",
                "false": "这个位置用词正确,应当保留原词",
            },
        }
        if report["candidates"]:
            criteria = {c["word"]: describe_candidate(c, report) for c in report["candidates"]}
            criteria[NONE_OPTION] = NONE_DESCRIPTION
            questions[f"replacement_{i}"] = {
                "type": "choice",
                "instructions": f"如果 `positions[{i}]` 这个位置应当被替换,换成下面哪个候选词最合适?"
                                f"候选词按 `positions[{i}].replacement_candidates` 的语料共现打分排序;"
                                f"如果它们都配不上这个位置,选 {NONE_OPTION}。",
                "criteria": criteria,
            }
    return questions


# ---------------------------------------------------------------------------
# 第三步:组合答案,得到纠正后的句子
# ---------------------------------------------------------------------------

def read_answers(response):
    """把 SDK 的 response 拍平成 ``{问题 id: {noul / choice / confidence / probabilities}}``。

    拍平之后决策逻辑只吃普通 dict,不依赖 SDK,测试里可以喂构造的答案。
    """
    answers = {}
    for name, answer in response.answers.items():
        answers[name] = {
            "type": getattr(answer, "type", None),
            "noul": getattr(answer, "noul", None),
            "choice": getattr(answer, "choice", None),
            "confidence": getattr(answer, "confidence", None),
            "probabilities": dict(getattr(answer, "probabilities", None) or {}),
        }
    return answers


def decide(report, answers, threshold):
    """一个位置的结论。

    ``replace_prob`` 是 jev 给的「应当替换」概率;达到 ``threshold`` 才换,换什么取
    choice 的 ``choice``;候选都不合适(none_of_these)或无候选时只报「需要人工确认」,
    不硬猜一个词。
    """
    replace_prob = answers.get(f"replace_{report['index']}", {}).get("noul")
    choice = answers.get(f"replacement_{report['index']}", {})

    decision = {
        "index": report["index"],
        "word": report["word"],
        "replace_prob": replace_prob,
        "replacement": None,
        "replacement_confidence": choice.get("confidence") if choice else None,
        "replacement_probabilities": choice.get("probabilities") if choice else {},
        "verdict": "unknown",
    }
    if replace_prob is None:                      # 模型没给这个位置的答案
        return decision
    if replace_prob < threshold:
        decision["verdict"] = "keep"
        return decision
    picked = choice.get("choice") if choice else None
    if picked and picked != NONE_OPTION:
        decision["verdict"] = "replace"
        decision["replacement"] = picked
    else:
        # 该位置有问题,但语料给不出合适的候选词,交给人工,不瞎猜
        decision["verdict"] = "replace_no_candidate"
    return decision


def apply_decisions(words, decisions):
    """按结论拼出纠正后的句子,并返回改动清单。"""
    corrected = list(words)
    changes = []
    for decision in decisions:
        if decision["verdict"] == "replace":
            corrected[decision["index"]] = decision["replacement"]
            changes.append({"index": decision["index"], "from": decision["word"],
                            "to": decision["replacement"],
                            "replace_prob": decision["replace_prob"]})
        elif decision["verdict"] == "replace_no_candidate":
            changes.append({"index": decision["index"], "from": decision["word"],
                            "to": None, "replace_prob": decision["replace_prob"]})
    return corrected, changes


def analyse_sample(words, stats, top_k=DEFAULT_TOP_K, context_top=DEFAULT_CONTEXT_TOP):
    """第一步的入口:检索 + 组装请求。返回 ``(reports, state, questions)``。"""
    lookup = build_lookup(stats)
    reports = analyse(words, lookup, top_k=top_k, context_top=context_top)
    state = build_state(words, reports, stats)
    questions = build_questions(reports, words)
    return reports, state, questions


# ---------------------------------------------------------------------------
# 调用 jev
# ---------------------------------------------------------------------------

def resolve_api_key(explicit=None):
    """按 --api-key > 环境变量 > Colab secret > 交互输入 的顺序找密钥。

    Colab 的 secret 走 ``google.colab.userdata``,这样 ``%run`` 与 ``!python`` 都能取到;
    两个都拿不到时给出设置方法,而不是抛一个看不懂的栈。
    """
    if explicit:
        return explicit
    if os.environ.get(API_KEY_ENV):
        return os.environ[API_KEY_ENV]
    try:
        from google.colab import userdata           # 只在 Colab 里有
        key = userdata.get(API_KEY_ENV)
        if key:
            return key
    except Exception:                               # 不在 Colab / 没这个 secret
        pass
    if sys.stdin.isatty():
        import getpass
        key = getpass.getpass(f"请输入 {API_KEY_ENV}: ").strip()
        if key:
            return key
    raise SystemExit(
        f"没找到 {API_KEY_ENV}。三种设法任选一种:\n"
        f"  1. 环境变量:export {API_KEY_ENV}=... (或在 cell 里 %env {API_KEY_ENV}=...)\n"
        f"  2. Colab:点左侧 🔑 图标,新建名为 {API_KEY_ENV} 的 secret\n"
        f"  3. 命令行:--api-key ...\n"
        f"密钥在 https://console.typesafe.ai/keys 创建。")


def call_jev(state, questions, model, api_key):
    """把 state + questions 发给 jev,返回 SDK 的 response。"""
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError:
        raise SystemExit("缺少 typesafe-sdk,先安装:pip install typesafe-sdk")

    with TypeSafeClient(api_key=api_key) as client:
        return client.system_one(state=state, questions=questions, model=model)


def resolve_stats_path(explicit=None):
    """找统计文件:--stats > 仓库默认位置 > ~/Downloads > 当前目录 > Colab 上传。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise SystemExit(f"统计文件 {path} 不存在;先用 "
                             f"python script/word_cooccurrence.py build 生成")
        return path

    candidates = [DEFAULT_STATS,
                  Path.home() / "Downloads" / "word_cooccurrence_CSL-Daily_train.json"]
    candidates += sorted(Path.cwd().glob("word_cooccurrence_*.json"))
    for path in candidates:
        if path.exists():
            return path

    uploaded = colab_upload_stats()
    if uploaded:
        return uploaded
    raise SystemExit(
        f"找不到统计文件,依次找过:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\n先用 python script/word_cooccurrence.py build 生成,"
          "或用 --stats 指定(/在 Colab 里把 word_cooccurrence_*.json 传上来后指定)。")


def colab_upload_stats():
    """在 Colab 里直接弹上传框;不在 Colab 或弹不出来就返回 None。"""
    try:
        from google.colab import files
    except ImportError:
        return None
    try:
        print("没找到统计文件,请上传 word_cooccurrence_*.json ...", file=sys.stderr)
        uploaded = files.upload()
    except Exception:                               # !python 子进程里没有 widget,退回手动上传
        return None
    for name in uploaded:
        if name.endswith(".json"):
            return Path(name)
    return None


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def relpath(path):
    """尽量显示成相对仓库根的路径,换机器后也读得懂。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def width(text):
    """按终端显示宽度算(中日韩全角字符占 2 列),让中英混排的表对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text, total, right=False):
    spaces = " " * max(total - width(text), 0)
    return spaces + text if right else text + spaces


def print_table(header, body):
    widths = [max(width(header[i]), *(width(row[i]) for row in body))
              for i in range(len(header))]
    print("    " + "  ".join(pad(header[i], widths[i]) for i in range(len(header))))
    for row in body:
        print("    " + "  ".join(pad(row[i], widths[i], right=i > 0) for i in range(len(row))))


def brief_neighbors(items, limit=3):
    if not items:
        return "-"
    return "、".join(f"{it['word']}({it['count']})" for it in items[:limit])


def print_step1(reports):
    print("\n第 1 步  逐词检索词频(纯代码,来自共现统计)")
    for r in reports:
        print(f"  下标 {r['index']}  词「{r['word']}」出现 {r['corpus_count']} 次"
              f"(占全语料 {r['corpus_share']:.3%})"
              f"  常见前词: {brief_neighbors(r['word_usual_preceders'])}"
              f"  常见后词: {brief_neighbors(r['word_usual_followers'])}")
        if r["candidates"]:
            listed = "、".join(f"{c['word']}({c['score']:.1%})" for c in r["candidates"])
            print(f"           候选替换(按左右邻搭配条件概率): {listed}")
        else:
            print("           候选替换: 无(左右邻词在语料里没有可用的搭配记录)")


def print_step2(decisions, threshold):
    print(f"\n第 2 步  jev 逐词判断(应替换概率 >= {threshold:g} 才替换)")
    body = []
    for d in decisions:
        prob = "-" if d["replace_prob"] is None else f"{d['replace_prob']:.3f}"
        if d["verdict"] == "replace":
            verdict, target = "替换", d["replacement"]
            extra = f"{d['replacement_confidence']:.2f}" if d["replacement_confidence"] is not None else "?"
        elif d["verdict"] == "replace_no_candidate":
            verdict, target, extra = "待人工确认", "(无合适候选)", "?"
        elif d["verdict"] == "keep":
            verdict, target, extra = "保留", "-", "-"
        else:
            verdict, target, extra = "无答案", "-", "-"
        body.append([str(d["index"]), d["word"], prob, verdict, target, extra])
    print_table(["下标", "原词", "应替换概率", "判定", "替换为", "选择置信度"], body)


def print_summary(words, corrected, changes):
    print(f"\n原句    {' '.join(words)}")
    print(f"纠正后  {' '.join(corrected)}")
    if not changes:
        print("没有需要改动的词。")
        return
    for change in changes:
        prob = f"{change['replace_prob']:.3f}"
        if change["to"]:
            print(f"  下标 {change['index']}: 「{change['from']}」-> 「{change['to']}」 (应替换概率 {prob})")
        else:
            print(f"  下标 {change['index']}: 「{change['from']}」语料候选都不合适,需人工确认 (应替换概率 {prob})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", help="样本的词,如: 他 每天 来 累 多")
    ap.add_argument("--sentence", help="整句(空格分隔);与位置参数等价,可整行粘 STM")
    ap.add_argument("--stats", type=Path, default=None,
                    help="word_cooccurrence.py build 产出的统计 JSON(默认自动找)")
    ap.add_argument("--model", default=os.environ.get("TYPESAFE_DEFAULT_MODEL", DEFAULT_MODEL),
                    help=f"jev 模型名(默认 {DEFAULT_MODEL})")
    ap.add_argument("--api-key", default=None, help="TypeSafe API key(默认读环境变量/Colab secret)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"应替换概率达到多少才真的替换(默认 {DEFAULT_THRESHOLD};"
                         f"换错代价高就调大)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"每个位置给 jev 的候选替换词个数(默认 {DEFAULT_TOP_K})")
    ap.add_argument("--context-top", type=int, default=DEFAULT_CONTEXT_TOP,
                    help=f"词频证据里每个方向列几个相邻词(默认 {DEFAULT_CONTEXT_TOP})")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印请求体(state + questions),不调用 API")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = ap.parse_args()

    words, stripped = parse_words(args.words, args.sentence)
    if not words:
        raise SystemExit("没有输入词。例:python script/word_correction_jev.py 他 每天 来 累 多")
    if stripped and not args.json:
        print("输入按整行 STM 处理,已丢掉前 5 列元信息。")

    stats_path = resolve_stats_path(args.stats)
    stats = load_stats(stats_path)
    reports, state, questions = analyse_sample(
        words, stats, top_k=args.top_k, context_top=args.context_top)

    if args.dry_run:
        print(json.dumps({"state": state, "model": args.model, "questions": questions},
                         ensure_ascii=False, indent=2))
        return 0

    api_key = resolve_api_key(args.api_key)
    response = call_jev(state, questions, args.model, api_key)
    answers = read_answers(response)

    decisions = [decide(report, answers, args.threshold) for report in reports]
    corrected, changes = apply_decisions(words, decisions)

    if args.json:
        print(json.dumps({
            "sentence": words,
            "corrected_sentence": corrected,
            "model": getattr(response, "model", args.model),
            "threshold": args.threshold,
            "stats_file": str(stats_path),
            "usage": {
                "input_tokens": getattr(getattr(response, "usage", None), "input_tokens", None),
                "output_tokens": getattr(getattr(response, "usage", None), "output_tokens", None),
            },
            # positions 直接用发给 jev 的那份 state,再挂上判定结果,省得两处各有一套字段名
            "positions": [{**position, "decision": {k: v for k, v in decision.items() if k != "index"}}
                          for position, decision in zip(state["positions"], decisions)],
            "changes": changes,
        }, ensure_ascii=False, indent=1))
        return 0

    meta = stats["meta"]
    print(f"统计文件 {relpath(stats_path)}")
    print(f"  样本 {meta.get('num_samples')} 条 | 词表 {meta.get('vocab_size')} "
          f"| 总词数 {meta.get('num_tokens')} | 来源 {meta.get('source_files')}")
    print(f"原句    {' '.join(words)}")
    print(f"请求    {len(questions)} 个问题({len(reports)} 个 noul + "
          f"{len(questions) - len(reports)} 个 choice),模型 {args.model},一次调用返回")
    print_step1(reports)
    print_step2(decisions, args.threshold)
    print_summary(words, corrected, changes)
    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"\ntoken: 输入 {usage.input_tokens} / 输出 {usage.output_tokens}"
              f"(jev 返回于 {datetime.now(timezone.utc).isoformat(timespec='seconds')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
