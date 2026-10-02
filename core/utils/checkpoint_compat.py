# -*- encoding: utf-8 -*-
"""checkpoint 与模型的结构差异判定(纯标准库,不依赖 torch)。

加载别人的权重时 key 不可能每次都严丝合缝。放任 ``strict=False`` 什么都不管会
埋雷,一律 ``strict=True`` 又会把「上游多带 6 个死张量」这种无害差异也变成加载
失败。这里只对**可证明不影响前向**的键放行,其余照旧拒绝。

目前需要放行的两类键都来自 SlowFast(上游 immc-lab/OpenCSLR 与本仓库 04ede88
之间的差异):

1. ``temporal_module_container.module_list.0.fc.*``
   ``TemporalSlowFastConv1D`` 自带的一个 ``nn.ModuleList``。上游 forward 从不读它
   (真正生效的是 ``TemporalSlowFastFuse.fc``,由 ``temporal_model`` 写入),所以
   缺了它只是少 6 个永远拿不到梯度的张量。本地曾把它删掉,为对齐上游又加了回来,
   因此两边 checkpoint 会相差这几个键。

2. ``temporal_module_container.module_list.1.conv1d.*``
   ``temporal_model`` 把同一个 conv1d 对象又注册了一遍,state_dict 里因此有一组
   指向**同一批参数**的别名键。别名键在 checkpoint 里缺失时值也是对的:
   ``load_state_dict`` 是原地 ``copy_``,主键路径加载后,别名指向的同一张量自然
   就是新值 —— 所以只需忽略,不需要额外搬运。

除此之外的差异一律视为真不匹配。
"""

#: SlowFast 时序容器里主 conv1d 的键前缀(Container.module_list[0])
PRIMARY_CONV1D_PREFIX = "temporal_module_container.module_list.0."

#: temporal_model 里重复注册出来的别名键前缀(Container.module_list[1])
ALIAS_CONV1D_PREFIX = "temporal_module_container.module_list.1.conv1d."

#: TemporalSlowFastConv1D 里从不参与前向的外层 fc
DEAD_SLOWFAST_FC_PREFIX = "temporal_module_container.module_list.0.fc."

#: BatchNorm 的计数器缓冲;缺失时只影响计数,不影响任何计算
_BUFFER_SUFFIXES = (".num_batches_tracked",)


def alias_source_key(key):
    """别名键 -> 对应的主键;不是别名键时返回 None。"""
    if key.startswith(ALIAS_CONV1D_PREFIX):
        return PRIMARY_CONV1D_PREFIX + key[len(ALIAS_CONV1D_PREFIX):]
    return None


def is_inert_missing(key):
    """模型需要、但 checkpoint 里没有 —— 可证明不影响计算的键。"""
    if alias_source_key(key) is not None:
        return True  # 同一批参数,主键加载后别名自动同步
    if key.startswith(DEAD_SLOWFAST_FC_PREFIX):
        return True  # 从不参与前向
    return key.endswith(_BUFFER_SUFFIXES)


def is_inert_unexpected(key):
    """checkpoint 里有、模型不需要 —— 可证明不影响计算的键。"""
    if alias_source_key(key) is not None:
        return True
    return key.startswith(DEAD_SLOWFAST_FC_PREFIX)


def partition_key_diff(missing_keys=(), unexpected_keys=()):
    """把 state_dict 的差异分成「可忽略」与「必须报错」两类。

    Args:
        missing_keys: 模型有、checkpoint 没有的键。
        unexpected_keys: checkpoint 有、模型没有的键。

    Returns:
        dict: ``ignorable_missing`` / ``fatal_missing`` /
        ``ignorable_unexpected`` / ``fatal_unexpected`` 四个已排序列表。
    """
    missing = sorted(str(key) for key in (missing_keys or ()))
    unexpected = sorted(str(key) for key in (unexpected_keys or ()))
    return {
        "ignorable_missing": [key for key in missing if is_inert_missing(key)],
        "fatal_missing": [key for key in missing if not is_inert_missing(key)],
        "ignorable_unexpected": [key for key in unexpected if is_inert_unexpected(key)],
        "fatal_unexpected": [key for key in unexpected if not is_inert_unexpected(key)],
    }


def describe_diff(report, limit=8):
    """把判定结果压成一条可直接给用户看的说明。"""
    def _preview(keys):
        head = keys[:limit]
        return ", ".join(head) + (" ...(共 %d 个)" % len(keys) if len(keys) > limit else "")

    parts = []
    if report["fatal_missing"]:
        parts.append("模型需要但权重里没有: " + _preview(report["fatal_missing"]))
    if report["fatal_unexpected"]:
        parts.append("权重里有但模型不需要: " + _preview(report["fatal_unexpected"]))
    return "; ".join(parts)


def describe_ignorable(report, limit=4):
    """可忽略差异的摘要(写日志用)。"""
    pieces = []
    if report["ignorable_missing"]:
        pieces.append("缺少 %d 个(别名/死参数/BatchNorm 计数)" % len(report["ignorable_missing"]))
    if report["ignorable_unexpected"]:
        pieces.append("多出 %d 个(别名/死参数)" % len(report["ignorable_unexpected"]))
    return ";".join(pieces)
