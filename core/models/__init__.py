# -*- coding: utf-8 -*-
"""模型层框架与注册表。

这个文件收拢了模型层的全部框架代码,具体模型各占一个文件:

    本文件                          Keys / require —— data dict 的键约定
                                    Container / SignLanguageModel —— 四容器组装契约
                                    register_model / get_model_builder —— 注册表

    tlp.py / sen.py / vac.py        各模型的 build_* 构建函数,以及该模型自己的
    corrnet.py / slowfast.py        损失组装(如 SENLoss)和解码覆盖(如 SlowFastDecoder)

    通用积木在 core/modules/ 下(ResNet、BiLSTM、tconv,以及最小单元损失
    CTCLoss / SeqKD 和通用 Decoder)。

新增模型:在 models/ 下新建一个文件,用 ``@register_model("名字")`` 装饰构建
函数,并在本文件末尾加一行 import;config 里写 ``model: 名字`` 即可。
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List


# ============================================================================
# 数据字典的 key 约定
# ============================================================================

class Keys(object):
    """data dict 的全部 key,按生产阶段分组。

    所有模块通过 Keys.* 读写 data dict,不要散落字符串字面量,
    避免 key 拼写错误只能到运行期才暴露。新增 key 时先在这里登记。
    """

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
    LOSS_LIFTPOOL_U = "loss_LiftPool_u"  # TLP 辅助损失(updater)
    LOSS_LIFTPOOL_P = "loss_LiftPool_p"  # TLP 辅助损失(predictor)
    HIDDEN = "hidden"                 # BiLSTM 隐状态

    # ---- 损失 / 解码输出 ----
    LOSS = "loss"
    TOTAL_LOSS = "total_loss"
    RECOGNIZED_SENTS = "recognized_sents"


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


# ============================================================================
# 模型基类
# ============================================================================

class Container(nn.Module):
    """容器模块,将同阶段的多个子模块串联执行。

    每个子模块的 forward 必须返回 dict,容器会依次调用并原地更新 data 字典。
    """

    def __init__(self, modules: List[nn.Module]):
        """初始化容器。

        Args:
            modules: 同阶段的子模块列表,将按顺序执行。
        """
        super(Container, self).__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """依次调用所有子模块,每个子模块的输出 dict 原地更新到 data 中。

        Args:
            data: 数据字典,包含当前阶段所需的全部键值。

        Returns:
            更新后的数据字典,包含所有子模块新增的键值。

        Raises:
            TypeError: 子模块未返回 dict 类型。
        """
        for module in self.module_list:
            output = module(data)
            if not isinstance(output, dict):
                raise TypeError(
                    "Container 中的 {} 必须返回 dict,实际返回 {}".format(
                        module.__class__.__name__, type(output).__name__))
            data.update(output)
        return data


class SignLanguageModel(nn.Module):
    """CSLR 模型基类,由四个 Container 按固定顺序组装。

    执行顺序:
        1. spatial_module_container  — 空间模块(帧级特征提取)
        2. temporal_module_container — 时序模块(时序建模与分类)
        3. loss_module_container     — 损失模块(CTC 损失等)
        4. decoder                   — 解码器(将输出转为文本)
    """

    def __init__(self, spatial_module_container: Container, temporal_module_container: Container,
                 loss_module_container: Container, decoder):
        """初始化模型。

        Args:
            spatial_module_container:  空间模块容器。
            temporal_module_container: 时序模块容器。
            loss_module_container:     损失模块容器。
            decoder:                   解码器,需实现返回 dict 的 forward 方法。
        """
        super().__init__()
        self.spatial_module_container = spatial_module_container
        self.temporal_module_container = temporal_module_container
        self.loss_module_container = loss_module_container
        self.decoder = decoder
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        """反向传播钩子,将梯度中的 NaN 置 0,防止 NaN 传播。"""
        for g in grad_input:
            if g is not None:
                g[g != g] = 0

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """按顺序执行四个容器,各容器原地更新 data 字典。

        Args:
            data: 输入数据字典,至少包含 vid、vid_lgt 等键。

        Returns:
            最终的数据字典,包含损失和/或解码结果。

        Raises:
            TypeError: decoder 未返回 dict 类型。
        """
        # 各容器原地更新 data 并返回同一字典,无需重复 update
        self.spatial_module_container(data)
        self.temporal_module_container(data)
        self.loss_module_container(data)
        output = self.decoder(data)
        if not isinstance(output, dict):
            raise TypeError(
                "decoder {} 必须返回 dict,实际返回 {}".format(
                    self.decoder.__class__.__name__, type(output).__name__))
        data.update(output)
        return data


# ============================================================================
# 构建函数注册表
# ============================================================================

MODEL_BUILDERS = {}


def register_model(name):
    """注册模型构建函数的装饰器。

    Args:
        name: 模型注册名(config 中 ``model:`` 字段使用的名字)。

    Returns:
        装饰器函数。
    """

    def decorator(fn):
        if name in MODEL_BUILDERS:
            raise ValueError(f"Model {name!r} is already registered.")
        MODEL_BUILDERS[name] = fn
        return fn

    return decorator


def get_model_builder(name):
    """按注册名获取模型构建函数。

    Args:
        name: 模型注册名。

    Returns:
        callable: 模型构建函数。

    Raises:
        ValueError: 注册名不存在时抛出,并列出全部可用模型。
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_BUILDERS)}."
        )
    return MODEL_BUILDERS[name]


# ============================================================================
# 导入具体模型,触发各自的 @register_model 注册
# 必须放在 Keys / Container / 注册表定义之后:具体模型会回头 from models import
# 这些名字,此时本模块已部分初始化,上面的定义已可见。
# ============================================================================

from . import corrnet, sen, slowfast, tlp, vac  # noqa: E402,F401
