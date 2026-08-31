# -*- coding: utf-8 -*-
"""
ctcdecode_py —— parlance/ctcdecode 的纯 Python 移植版
======================================================

原项目（https://github.com/parlance/ctcdecode）的 CTC 束搜索解码器由 C++ 编写，
编译依赖 PyTorch C++ 扩展 / Boost / OpenFST / KenLM，门槛较高。本文件用纯
Python 1:1 移植其核心算法，**无需任何编译**：

* ``PathTrie``                <- path_trie.cpp（去掉了 OpenFST 词典约束）
* ``DecoderState``            <- ctc_beam_search_decoder.cpp
* ``get_pruned_log_probs`` 等 <- decoder_utils.cpp
* ``CTCBeamDecoder``          <- ctcdecode/__init__.py 的 PyTorch 封装
* ``OnlineCTCBeamDecoder``    <- 同上的流式（分块）解码封装

与原版的差异
------------
1. 只依赖 Python 标准库。若环境中装有 torch，则输入/输出与原 API 完全一致
   （``torch.Tensor``）；没有 torch 时自动退回 numpy 数组或纯 list。
2. KenLM 是 C++ 库，纯 Python 无法加载 ``.bin/.arpa`` 模型，因此
   ``model_path`` 参数不可用。如需外接语言模型，可通过 ``scorer=`` 传入一个
   纯 Python 对象，需实现如下协议（与 C++ Scorer 对应）::

       scorer.alpha                      -> float，LM 权重
       scorer.beta                       -> float，词数奖励
       scorer.is_character_based()       -> bool
       scorer.make_ngram(prefix)         -> 由 PathTrie 节点构造 n-gram（list[str]）
       scorer.get_log_cond_prob(ngram)   -> float，log 条件概率
       scorer.get_sent_log_prob(words)   -> float，整句 log 概率
       scorer.split_labels(token_ids)    -> list[str]，把 token 序列切成词

   prefix 节点可用 ``prefix.get_path_vec()`` 得到 ``(token_ids, timesteps)``。
3. C++ 内部用 float32 存中间概率，本移植用 float64，结果有 1e-6 量级差异
   （更精确）；另外原 C++ 版输出张量的 padding 是未初始化内存，本版统一填 0。
4. ``num_processes`` 仅为 API 兼容保留，纯 Python 下串行解码。
5. 速度比 C++ 慢约 1~2 个数量级。长序列/大批量建议调小 ``beam_width`` 和
   ``cutoff_top_n``，或仅用于中小规模任务与调试。

用法（与原包一致）
------------------
>>> from ctcdecode_py import CTCBeamDecoder
>>> decoder = CTCBeamDecoder(labels, beam_width=100, cutoff_top_n=40,
...                          cutoff_prob=1.0, blank_id=0, log_probs_input=True)
>>> beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs, seq_lens)

也可以把本文件重命名为 ``ctcdecode.py`` 直接替换原包（API 签名一致）。
"""

from __future__ import annotations

import math

NEG_INF = float("-inf")
# 与 C++ 端 std::numeric_limits<float>::min() 相同，用于避免 log(0)
FLT_MIN = 1.1754943508222875e-38

try:  # 输出格式优先与原版一致（torch.Tensor）
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    import numpy as np

    _HAS_NP = True
except ImportError:  # pragma: no cover
    _HAS_NP = False


# ---------------------------------------------------------------------------
# decoder_utils.cpp
# ---------------------------------------------------------------------------


def log_sum_exp(x, y):
    """log(exp(x) + exp(y))，-inf 安全，与 C++ 版 log_sum_exp 等价。"""
    if x <= NEG_INF:
        return y
    if y <= NEG_INF:
        return x
    xmax = x if x > y else y
    return math.log(math.exp(x - xmax) + math.exp(y - xmax)) + xmax


def _to_log(p, log_input):
    """按 log_input 把输入转成 log 概率（C++ 中 log(0) 得 -inf，这里做同样处理）。"""
    if log_input:
        return p
    return math.log(p) if p > 0.0 else NEG_INF


def get_pruned_log_probs(prob_step, cutoff_prob, cutoff_top_n, log_input):
    """对应 decoder_utils.cpp 的 get_pruned_log_probs。

    按 cutoff_prob（累计概率）与 cutoff_top_n（个数）裁剪每帧的词表，
    返回 [(char_id, log_prob), ...]。
    """
    cutoff_len = len(prob_step)
    log_cutoff_prob = math.log(cutoff_prob)
    if log_cutoff_prob < 0.0 or cutoff_top_n < cutoff_len:
        # 按概率降序（对应 C++ pair_comp_second_rev）
        prob_idx = sorted(range(len(prob_step)), key=lambda i: -prob_step[i])
        if log_cutoff_prob < 0.0:
            # 注意：此处刻意保持与原版一致的行为 —— cum_prob 在 log 域从 0.0
            # 开始累加，再与线性的 cutoff_prob 比较（原仓库即如此实现）。
            cum_prob = 0.0
            cutoff_len = 0
            for i in prob_idx:
                cum_prob = log_sum_exp(cum_prob, _to_log(prob_step[i], log_input))
                cutoff_len += 1
                if cum_prob >= cutoff_prob or cutoff_len >= cutoff_top_n:
                    break
        else:
            cutoff_len = cutoff_top_n
        prob_idx = prob_idx[:cutoff_len]
    else:
        prob_idx = list(range(len(prob_step)))

    log_prob_idx = []
    for i in prob_idx:
        p = prob_step[i]
        lp = p if log_input else math.log(p + FLT_MIN)
        log_prob_idx.append((i, lp))
    return log_prob_idx


def _prefix_key(p):
    """对应 prefix_compare：score 降序，并列时 character 升序。"""
    return (-p.score, p.character)


def get_beam_search_result(prefixes, beam_size):
    """对应 decoder_utils.cpp 的 get_beam_search_result。

    返回 [(score, tokens, timesteps), ...]，score = -approx_ctc（越小越好）。
    """
    space_prefixes = list(prefixes[:beam_size])
    space_prefixes.sort(key=_prefix_key)
    results = []
    for p in space_prefixes:
        tokens, timesteps = p.get_path_vec()
        results.append((-p.approx_ctc, tokens, timesteps))
    return results


# ---------------------------------------------------------------------------
# path_trie.cpp
# ---------------------------------------------------------------------------


class PathTrie:
    """前缀树节点，对应 path_trie.h/.cpp 的 PathTrie（不含 OpenFST 词典）。

    log_prob_b_prev / log_prob_nb_prev: 上一时刻以 blank / 非 blank 结尾的 log 概率
    log_prob_b_cur  / log_prob_nb_cur : 当前时刻的累积量
    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev)
    """

    ROOT = -1

    __slots__ = (
        "log_prob_b_prev",
        "log_prob_nb_prev",
        "log_prob_b_cur",
        "log_prob_nb_cur",
        "log_prob_c",
        "score",
        "approx_ctc",
        "character",
        "timestep",
        "parent",
        "exists",
        "children",
    )

    def __init__(self):
        self.log_prob_b_prev = NEG_INF
        self.log_prob_nb_prev = NEG_INF
        self.log_prob_b_cur = NEG_INF
        self.log_prob_nb_cur = NEG_INF
        self.log_prob_c = NEG_INF
        self.score = NEG_INF
        self.approx_ctc = 0.0
        self.character = self.ROOT
        self.timestep = 0
        self.parent = None
        self.exists = True
        self.children = {}  # char_id -> PathTrie

    def is_empty(self):
        return self.character == self.ROOT

    def get_path_trie(self, new_char, new_timestep, cur_log_prob_c):
        """获取（或新建）追加了 new_char 的子节点，对应 C++ get_path_trie。"""
        child = self.children.get(new_char)
        if child is not None:
            if child.log_prob_c < cur_log_prob_c:
                child.log_prob_c = cur_log_prob_c
                child.timestep = new_timestep
            if not child.exists:
                # 节点曾被剪枝，复活并清零概率
                child.exists = True
                child.log_prob_b_prev = NEG_INF
                child.log_prob_nb_prev = NEG_INF
                child.log_prob_b_cur = NEG_INF
                child.log_prob_nb_cur = NEG_INF
            return child
        new_path = PathTrie()
        new_path.character = new_char
        new_path.timestep = new_timestep
        new_path.parent = self
        new_path.log_prob_c = cur_log_prob_c
        self.children[new_char] = new_path
        return new_path

    def get_path_vec(self):
        """返回 (tokens, timesteps)：从根到当前节点的字符 id 与对应时间步。"""
        output = []
        timesteps = []
        node = self
        while node.character != self.ROOT and node.parent is not None:
            output.append(node.character)
            timesteps.append(node.timestep)
            node = node.parent
        output.reverse()
        timesteps.reverse()
        return output, timesteps

    def iterate_to_vec(self, output):
        """cur -> prev 滚动，计算 score，并把所有存活节点收集到 output。

        用显式栈代替 C++ 的递归，避免长序列时超过 Python 递归深度。
        """
        stack = [self]
        while stack:
            node = stack.pop()
            if node.exists:
                node.log_prob_b_prev = node.log_prob_b_cur
                node.log_prob_nb_prev = node.log_prob_nb_cur
                node.log_prob_b_cur = NEG_INF
                node.log_prob_nb_cur = NEG_INF
                node.score = log_sum_exp(node.log_prob_b_prev, node.log_prob_nb_prev)
                output.append(node)
            if node.children:
                stack.extend(node.children.values())

    def remove(self):
        """标记删除；若是叶子则从父节点摘除，并向上级联清理空死节点。"""
        self.exists = False
        node = self
        while not node.children and node.parent is not None:
            parent = node.parent
            del parent.children[node.character]
            node = parent
            if node.children or node.exists:
                break


# ---------------------------------------------------------------------------
# ctc_beam_search_decoder.cpp
# ---------------------------------------------------------------------------


class DecoderState:
    """对应 ctc_beam_search_decoder.cpp 的 DecoderState（纯 Python，无 KenLM/FST）。

    参数含义与 CTCBeamDecoder 相同；ext_scorer 为可选的 Python 语言模型对象。
    """

    def __init__(
        self,
        vocabulary,
        beam_size,
        cutoff_prob,
        cutoff_top_n,
        blank_id,
        log_input,
        ext_scorer=None,
    ):
        self.abs_time_step = 0
        self.beam_size = beam_size
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n
        self.blank_id = blank_id
        self.log_input = log_input
        self.vocabulary = list(vocabulary)
        self.ext_scorer = ext_scorer
        self.space_id = self.vocabulary.index(" ") if " " in self.vocabulary else -2

        # 初始化前缀树根节点
        self.root = PathTrie()
        self.root.score = 0.0
        self.root.log_prob_b_prev = 0.0
        self.prefixes = [self.root]

    def next(self, probs_seq):
        """送入一段 probs（T x V 的 list），推进束搜索。可多次调用（流式）。"""
        num_labels = len(self.vocabulary)
        for prob in probs_seq:
            if len(prob) != num_labels:
                raise ValueError(
                    "The shape of probs_seq does not match with the shape of the vocabulary"
                )

            min_cutoff = NEG_INF
            full_beam = False
            if self.ext_scorer is not None:
                num_prefixes = min(len(self.prefixes), self.beam_size)
                self.prefixes[:num_prefixes] = sorted(
                    self.prefixes[:num_prefixes], key=_prefix_key
                )
                blank_prob = _to_log(prob[self.blank_id], self.log_input)
                min_cutoff = (
                    self.prefixes[num_prefixes - 1].score
                    + blank_prob
                    - max(0.0, self.ext_scorer.beta)
                )
                full_beam = num_prefixes == self.beam_size

            log_prob_idx = get_pruned_log_probs(
                prob, self.cutoff_prob, self.cutoff_top_n, self.log_input
            )

            # 遍历字符
            for c, log_prob_c in log_prob_idx:
                # 遍历当前 beam 中的前缀
                for i, prefix in enumerate(self.prefixes):
                    if i >= self.beam_size:
                        break
                    if full_beam and log_prob_c + prefix.score < min_cutoff:
                        break
                    # blank：概率并入当前前缀的 blank 结尾
                    if c == self.blank_id:
                        prefix.log_prob_b_cur = log_sum_exp(
                            prefix.log_prob_b_cur, log_prob_c + prefix.score
                        )
                        continue
                    # 重复字符：不产生新字符，概率留在原前缀上
                    if c == prefix.character:
                        prefix.log_prob_nb_cur = log_sum_exp(
                            prefix.log_prob_nb_cur,
                            log_prob_c + prefix.log_prob_nb_prev,
                        )
                    # 扩展得到新前缀
                    prefix_new = prefix.get_path_trie(c, self.abs_time_step, log_prob_c)
                    if prefix_new is not None:
                        log_p = NEG_INF
                        if c == prefix.character and prefix.log_prob_b_prev > NEG_INF:
                            # blank 隔开的重复字符 -> 真正的新字符
                            log_p = log_prob_c + prefix.log_prob_b_prev
                        elif c != prefix.character:
                            log_p = log_prob_c + prefix.score

                        # 语言模型打分（需要外部 Python scorer）
                        scorer = self.ext_scorer
                        if scorer is not None and (
                            c == self.space_id or scorer.is_character_based()
                        ):
                            if scorer.is_character_based():
                                prefix_to_score = prefix_new
                            else:
                                prefix_to_score = prefix  # 空格本身不打分
                            ngram = scorer.make_ngram(prefix_to_score)
                            log_p += scorer.get_log_cond_prob(ngram) * scorer.alpha
                            log_p += scorer.beta
                        prefix_new.log_prob_nb_cur = log_sum_exp(
                            prefix_new.log_prob_nb_cur, log_p
                        )

            # 滚动概率并收集存活前缀
            self.prefixes = []
            self.root.iterate_to_vec(self.prefixes)

            # 只保留 top beam_size
            if len(self.prefixes) >= self.beam_size:
                self.prefixes.sort(key=_prefix_key)
                for p in self.prefixes[self.beam_size :]:
                    p.remove()
                del self.prefixes[self.beam_size :]

            self.abs_time_step += 1

    def decode(self):
        """结束解码，返回 [(score, tokens, timesteps), ...]（按得分从好到差）。"""
        prefixes_copy = list(self.prefixes)
        scores = {p: p.score for p in prefixes_copy}

        scorer = self.ext_scorer
        # 给最后一个不以空格结尾的词补 LM 分
        if scorer is not None and not scorer.is_character_based():
            for prefix in prefixes_copy[: self.beam_size]:
                if not prefix.is_empty() and prefix.character != self.space_id:
                    s = scorer.get_log_cond_prob(scorer.make_ngram(prefix)) * scorer.alpha
                    s += scorer.beta
                    scores[prefix] += s

        num_prefixes = min(len(prefixes_copy), self.beam_size)
        prefixes_copy[:num_prefixes] = sorted(
            prefixes_copy[:num_prefixes], key=lambda p: (-scores[p], p.character)
        )

        # 计算近似的纯 CTC 分数作为返回分数（不影响返回顺序，与原版一致）
        for prefix in prefixes_copy[: self.beam_size]:
            approx_ctc = scores[prefix]
            if scorer is not None:
                output, _ = prefix.get_path_vec()
                words = scorer.split_labels(output)
                approx_ctc -= len(output) * scorer.beta  # 去掉词插入奖励
                approx_ctc -= scorer.get_sent_log_prob(words) * scorer.alpha  # 去掉 LM 分
            prefix.approx_ctc = approx_ctc

        return get_beam_search_result(prefixes_copy, self.beam_size)


def ctc_beam_search_decoder(
    probs_seq,
    vocabulary,
    beam_size,
    cutoff_prob,
    cutoff_top_n,
    blank_id,
    log_input,
    ext_scorer=None,
):
    """单条序列的束搜索解码，对应 C++ ctc_beam_search_decoder。

    probs_seq: T x V 的 list（概率或 log 概率，由 log_input 指明）。
    返回 [(score, tokens, timesteps), ...]，score 为负 log 概率（越小越好）。
    """
    state = DecoderState(
        vocabulary, beam_size, cutoff_prob, cutoff_top_n, blank_id, log_input, ext_scorer
    )
    state.next(probs_seq)
    return state.decode()


def ctc_beam_search_decoder_batch(
    probs_split,
    vocabulary,
    beam_size,
    num_processes,
    cutoff_prob,
    cutoff_top_n,
    blank_id,
    log_input,
    ext_scorer=None,
):
    """批量解码，对应 C++ ctc_beam_search_decoder_batch。

    num_processes 仅为 API 兼容保留（纯 Python 串行执行）。
    """
    if num_processes <= 0:
        raise ValueError("num_processes must be nonnegative!")
    return [
        ctc_beam_search_decoder(
            probs_seq,
            vocabulary,
            beam_size,
            cutoff_prob,
            cutoff_top_n,
            blank_id,
            log_input,
            ext_scorer,
        )
        for probs_seq in probs_split
    ]


# ---------------------------------------------------------------------------
# 输入/输出转换辅助
# ---------------------------------------------------------------------------


def _to_3d_list(probs):
    """把 (batch, T, V) 的 torch.Tensor / np.ndarray / list 转成 list。"""
    if _HAS_TORCH and isinstance(probs, torch.Tensor):
        return probs.detach().cpu().float().tolist()
    if _HAS_NP and isinstance(probs, np.ndarray):
        return probs.astype(float).tolist()
    return [[[float(v) for v in step] for step in seq] for seq in probs]


def _to_1d_list(seq_lens):
    if _HAS_TORCH and isinstance(seq_lens, torch.Tensor):
        return seq_lens.detach().cpu().int().tolist()
    if _HAS_NP and isinstance(seq_lens, np.ndarray):
        return seq_lens.astype(int).tolist()
    return [int(x) for x in seq_lens]


def _make_outputs(output, scores, timesteps, out_lens):
    """优先返回 torch.Tensor（与原版一致），否则 numpy，最后退回 list。"""
    if _HAS_TORCH:
        return (
            torch.tensor(output, dtype=torch.int32),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(timesteps, dtype=torch.int32),
            torch.tensor(out_lens, dtype=torch.int32),
        )
    if _HAS_NP:
        return (
            np.array(output, dtype=np.int32),
            np.array(scores, dtype=np.float32),
            np.array(timesteps, dtype=np.int32),
            np.array(out_lens, dtype=np.int32),
        )
    return output, scores, timesteps, out_lens


# ---------------------------------------------------------------------------
# ctcdecode/__init__.py 的封装
# ---------------------------------------------------------------------------


class CTCBeamDecoder(object):
    """DeepSpeech PaddlePaddle 束搜索解码器的纯 Python 封装（API 与原版一致）。

    Args:
        labels (list): 模型输出对应的 token 表，顺序与模型输出一致。
        model_path: 原版用于加载 KenLM 模型；纯 Python 版不可用（请用 scorer=）。
        alpha (float): LM 概率权重，0 表示 LM 不生效。
        beta (float):  beam 内词数奖励。
        cutoff_top_n (int): 每帧只取概率最高的前 cutoff_top_n 个字符。
        cutoff_prob (float): 累计概率裁剪阈值，1.0 表示不裁剪。
        beam_width (int): beam 宽度，越大越准但越慢。
        num_processes (int): 仅为兼容保留（纯 Python 串行解码）。
        blank_id (int): CTC blank 的 id（通常为 0）。
        log_probs_input (bool): 输入已是 log 概率则为 True；若模型输出经过
            softmax（概率和为 1）则为 False。
        scorer: 可选的纯 Python 语言模型对象（协议见模块 docstring）。
    """

    def __init__(
        self,
        labels,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=False,
        scorer=None,
    ):
        if model_path:
            raise NotImplementedError(
                "纯 Python 版无法加载 KenLM 模型（%r）。不加 LM 请去掉 model_path；"
                "如需 LM，请通过 scorer= 传入纯 Python 实现的 scorer 对象。" % model_path
            )
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = scorer
        self._num_processes = num_processes
        self._labels = list(labels)
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        """对模型输出做束搜索并返回结果。

        Args:
            probs: (batch, num_timesteps, num_labels) 的 Tensor / ndarray / list。
            seq_lens: (batch,) 各样本的有效帧数，可选；缺省用 probs 的第 1 维。

        Returns:
            (beam_results, beam_scores, timesteps, out_lens)：
              beam_results: (batch, beam_width, num_timesteps) int，token id，0 填充
              beam_scores:  (batch, beam_width) float，负 log 概率，越小越好
              timesteps:    (batch, beam_width, num_timesteps) int，各输出字符
                            概率峰值所在的时间步（音字对齐用）
              out_lens:     (batch, beam_width) int，每条 beam 的实际长度
            装有 torch 时返回 torch.Tensor，否则 numpy 数组（均无则 list）。
        """
        probs_list = _to_3d_list(probs)
        batch_size = len(probs_list)
        max_seq_len = len(probs_list[0]) if batch_size else 0
        if seq_lens is None:
            seq_lens_list = [max_seq_len] * batch_size
        else:
            seq_lens_list = [min(l, max_seq_len) for l in _to_1d_list(seq_lens)]

        inputs = [probs_list[b][: seq_lens_list[b]] for b in range(batch_size)]
        batch_results = ctc_beam_search_decoder_batch(
            inputs,
            self._labels,
            self._beam_width,
            self._num_processes,
            self._cutoff_prob,
            self.cutoff_top_n,
            self._blank_id,
            self._log_probs,
            self._scorer,
        )

        output = [[[0] * max_seq_len for _ in range(self._beam_width)] for _ in range(batch_size)]
        timesteps = [[[0] * max_seq_len for _ in range(self._beam_width)] for _ in range(batch_size)]
        scores = [[0.0] * self._beam_width for _ in range(batch_size)]
        out_lens = [[0] * self._beam_width for _ in range(batch_size)]
        for b, results in enumerate(batch_results):
            for p, (score, tokens, ts) in enumerate(results):
                output[b][p][: len(tokens)] = tokens
                timesteps[b][p][: len(ts)] = ts
                scores[b][p] = score
                out_lens[b][p] = len(tokens)

        return _make_outputs(output, scores, timesteps, out_lens)

    # --- 以下为原版 LM 相关接口，纯 Python 版在 scorer 支持时透传 ---

    def character_based(self):
        return self._scorer.is_character_based() if self._scorer else None

    def max_order(self):
        return getattr(self._scorer, "get_max_order", lambda: None)() if self._scorer else None

    def dict_size(self):
        return getattr(self._scorer, "get_dict_size", lambda: None)() if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            self._scorer.alpha = alpha
            self._scorer.beta = beta


class OnlineCTCBeamDecoder(object):
    """流式（分块）束搜索解码器，对应原版 OnlineCTCBeamDecoder。

    与 CTCBeamDecoder 参数相同。用法：

    >>> decoder = OnlineCTCBeamDecoder(labels, beam_width=20, log_probs_input=True)
    >>> states = [decoder.create_state() for _ in range(batch_size)]
    >>> for chunk_probs, is_last in stream:  # chunk_probs: (batch, t, V)
    ...     out = decoder.decode(chunk_probs, states, [is_last] * batch_size)
    """

    def __init__(
        self,
        labels,
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=0,
        log_probs_input=False,
        scorer=None,
    ):
        if model_path:
            raise NotImplementedError(
                "纯 Python 版无法加载 KenLM 模型（%r），请通过 scorer= 传入 Python scorer。"
                % model_path
            )
        self._cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._scorer = scorer
        self._num_processes = num_processes
        self._labels = list(labels)
        self._num_labels = len(labels)
        self._blank_id = blank_id
        self._log_probs = 1 if log_probs_input else 0
        self._cutoff_prob = cutoff_prob

    def create_state(self):
        """新建一个解码状态（对应原版的 ctcdecode.DecoderState(decoder)）。

        纯 Python 对象，无需手动释放；一个状态对应一条音频，不要复用。
        """
        return DecoderState(
            self._labels,
            self._beam_width,
            self._cutoff_prob,
            self._cutoff_top_n,
            self._blank_id,
            self._log_probs,
            self._scorer,
        )

    def decode(self, probs, states, is_eos_s, seq_lens=None):
        """推进一批 chunk 的解码状态；is_eos_s=True 的样本返回最终结果。

        Args:
            probs: (batch, t, V) 本 chunk 的输出。
            states: 长度为 batch 的 DecoderState（由 create_state() 创建）。
            is_eos_s: 长度为 batch 的 bool，是否为最后一个 chunk。
            seq_lens: 可选，各样本在本 chunk 的有效帧数。

        Returns:
            (beam_results, beam_scores, timesteps, out_lens)，形状按本批
            最大 beam 数 / 最大序列长对齐，0 填充；未到 eos 的样本结果为空。
        """
        probs_list = _to_3d_list(probs)
        batch_size = len(probs_list)
        max_seq_len = len(probs_list[0]) if batch_size else 0
        if seq_lens is None:
            seq_lens_list = [max_seq_len] * batch_size
        else:
            seq_lens_list = [min(l, max_seq_len) for l in _to_1d_list(seq_lens)]

        results_per_sample = []
        for b in range(batch_size):
            state = states[b]
            if not isinstance(state, DecoderState):  # 兼容带 .state 属性的句柄
                state = state.state
            state.next(probs_list[b][: seq_lens_list[b]])
            results_per_sample.append(state.decode() if is_eos_s[b] else [])

        max_beams = max((len(r) for r in results_per_sample), default=0)
        max_tokens = max(
            (len(tokens) for r in results_per_sample for _, tokens, _ in r), default=0
        )
        output = [[[0] * max_tokens for _ in range(max_beams)] for _ in range(batch_size)]
        timesteps = [[[0] * max_tokens for _ in range(max_beams)] for _ in range(batch_size)]
        scores = [[0.0] * max_beams for _ in range(batch_size)]
        out_lens = [[0] * max_beams for _ in range(batch_size)]
        for b, results in enumerate(results_per_sample):
            for p, (score, tokens, ts) in enumerate(results):
                output[b][p][: len(tokens)] = tokens
                timesteps[b][p][: len(ts)] = ts
                scores[b][p] = score
                out_lens[b][p] = len(tokens)

        return _make_outputs(output, scores, timesteps, out_lens)

    def character_based(self):
        return self._scorer.is_character_based() if self._scorer else None

    def max_order(self):
        return getattr(self._scorer, "get_max_order", lambda: None)() if self._scorer else None

    def dict_size(self):
        return getattr(self._scorer, "get_dict_size", lambda: None)() if self._scorer else None

    @staticmethod
    def reset_state(state):
        """原版用于释放 C++ 状态；纯 Python 版无需操作（GC 自动回收）。"""
        pass
