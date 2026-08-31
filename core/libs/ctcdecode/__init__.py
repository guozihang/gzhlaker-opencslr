# -*- coding: utf-8 -*-
"""仓库内置的 CTC 束搜索解码器(纯 Python 移植版,无需编译安装)。

将 ctcdecode_py.py 中移植自 parlance/ctcdecode 的实现暴露为与原
ctcdecode 包一致的 API,用法::

    from libs.ctcdecode import CTCBeamDecoder

    decoder = CTCBeamDecoder(labels, beam_width=100, blank_id=0)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(probs, seq_lens)

与原 C++ 版的差异及使用说明见 ctcdecode_py 模块 docstring。
"""

from .ctcdecode_py import (
    CTCBeamDecoder,
    OnlineCTCBeamDecoder,
    ctc_beam_search_decoder,
    ctc_beam_search_decoder_batch,
)

__all__ = [
    "CTCBeamDecoder",
    "OnlineCTCBeamDecoder",
    "ctc_beam_search_decoder",
    "ctc_beam_search_decoder_batch",
]
