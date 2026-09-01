# -*- coding: utf-8 -*-
"""模型基类。

定义 Container 和 SignLanguageModel 两个核心基类。
Container 负责串联同阶段的多个子模块,SignLanguageModel 将四个 Container 按
"空间→时序→损失→解码" 的顺序组装成完整的 CSLR 模型。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List


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
