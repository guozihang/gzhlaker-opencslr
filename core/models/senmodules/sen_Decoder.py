"""SEN 模型的解码器模块。

将模型输出的序列 logits 解码为可读的手语词汇序列，
支持集束搜索（beam search）和贪婪搜索（max decoding）两种模式。
"""
import torch.nn as nn

from ..modules.decoder import Decode
from ..keys import Keys, require


class sen_Decoder(nn.Module):
    """SEN 解码器模块。

    作为 SEN 模型中的 decoder 使用，在推理阶段将模型输出的
    序列 logits 解码为手语词汇序列。

    Args:
        args: 配置字典，包含 decode_mode 和 num_classes
        gloss_dict: 词汇表字典，映射 gloss ID 到词汇
    """

    def __init__(self, args, gloss_dict):
        super(sen_Decoder, self).__init__()
        search_mode = args.get("decode_mode", "beam")
        self.decoder = Decode(gloss_dict, args["num_classes"], search_mode)

    def forward(self, data):
        """前向传播（解码）。

        Args:
            data: 包含 Keys.SEQUENCE_LOGITS 和 Keys.FEAT_LEN 的字典

        Returns:
            dict: 包含 Keys.RECOGNIZED_SENTS 的字典
        """
        require(data, Keys.SEQUENCE_LOGITS, Keys.FEAT_LEN, who="sen_Decoder")
        pred = None
        if not self.training:
            pred = self.decoder.decode(data[Keys.SEQUENCE_LOGITS], data[Keys.FEAT_LEN], batch_first=False, probs=False)
        return {
            Keys.RECOGNIZED_SENTS: pred
        }
