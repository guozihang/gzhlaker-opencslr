# -*- coding: utf-8 -*-
"""
模型数据字典的 key 常量与校验工具。

所有模块通过 Keys.* 读写 data dict,不要散落字符串字面量,
避免 key 拼写错误只能到运行期才暴露。新增 key 时先在这里登记。
"""


class Keys(object):
    """data dict 的全部 key,按生产阶段分组。"""

    # ---- 输入:由 pipeline(single.py)注入 ----
    VID = "vid"                       # 视频帧 (B, T, C, H, W) 或预提取帧特征
    VID_LGT = "vid_lgt"               # 原始视频帧长 (B,)
    LABEL = "label"                   # 标签索引 (B, L)
    LABEL_LGT = "label_lgt"           # 标签长度 (B,)
    INFO = "info"                     # 文件名信息,仅 eval 阶段

    # ---- 空间模块输出 ----
    FRAMEWISE_FEATURES = "framewise_features"   # 帧级特征 (C, T, B)

    # ---- 时序模块输出 ----
    VISUAL_FEAT = "visual_feat"       # 时序特征 (T, B, C);SlowFast 下为 list
    CONV_LOGITS = "conv_logits"       # 时序卷积分类输出;SlowFast 下为 list
    FEAT_LEN = "feat_len"             # 下采样后的特征长度 (B,)
    PREDICTIONS = "predictions"       # BiLSTM 输出
    SEQUENCE_LOGITS = "sequence_logits"  # 主分类输出;SlowFast 下为 list
    OUTPUT_FIRST = "output_first"     # SlowFast:主通路 logits
    LOSS_LIFTPOOL_U = "loss_LiftPool_u"  # TLP 辅助损失(updater)
    LOSS_LIFTPOOL_P = "loss_LiftPool_p"  # TLP 辅助损失(predictor)
    HIDDEN = "hidden"                 # BiLSTM 隐状态

    # ---- 损失 / 解码输出 ----
    LOSS = "loss"
    TOTAL_LOSS = "total_loss"
    RECOGNIZED_SENTS = "recognized_sents"
    CONV_SENTS = "conv_sents"


def require(data, *keys, who=""):
    """校验 data 中包含全部必需 key,缺失时立即抛错并列出当前可用 key。

    Args:
        data: 模型数据字典。
        *keys: 必需的 key(使用 Keys.* 常量)。
        who: 调用方模块名,仅用于错误信息。
    """
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(
            "{}缺少必需数据 key: {!r};当前可用: {!r}".format(
                (who + " ") if who else who, missing, sorted(data.keys())
            )
        )
    return data
