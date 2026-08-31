import os
import pdb
import time
import torch
from libs.ctcdecode import CTCBeamDecoder
import numpy as np
from itertools import groupby
import torch.nn.functional as F
from models.keys import Keys, require

"""SlowFast 解码器模块。

提供 CTC 解码功能，支持集束搜索（beam search）和贪婪搜索（max decoding）
两种模式，将模型输出的序列 logits 解码为手语词汇序列。
"""


class SlowFast_Decoder:
    """SlowFast 解码器。

    在推理阶段将模型输出的序列 logits 解码为手语词汇序列，
    支持主路径和卷积路径的独立解码。

    Args:
        args: 配置字典，包含 num_classes 等参数
        gloss_dict: 词汇表字典，映射 gloss ID 到词汇
    """

    def __init__(self, args, gloss_dict):
        super(SlowFast_Decoder, self).__init__()
        self.decoder = Decode(gloss_dict, args["num_classes"], 'beam')

    def __call__(self, data):
        """执行解码。

        Args:
            data: 包含 Keys.OUTPUT_FIRST, Keys.CONV_LOGITS, Keys.FEAT_LEN 的字典

        Returns:
            dict: 包含卷积路径和主路径解码结果的字典
        """

        require(data, Keys.OUTPUT_FIRST, Keys.CONV_LOGITS, Keys.FEAT_LEN, who="SlowFast_Decoder")
        pred = self.decoder.decode(data[Keys.OUTPUT_FIRST], data[Keys.FEAT_LEN], batch_first=False, probs=False)
        conv_pred = self.decoder.decode(data[Keys.CONV_LOGITS][0], data[Keys.FEAT_LEN], batch_first=False, probs=False)

        return {
            Keys.CONV_SENTS: conv_pred,
            Keys.RECOGNIZED_SENTS: pred,
        }

class Decode(object):
    """CTC 解码器基类。

    支持贪婪搜索（max decoding）和集束搜索（beam search）两种解码模式，
    将神经网络输出解码为词汇序列。

    Args:
        gloss_dict: 词汇表字典，映射 gloss ID 到词汇
        num_classes: 分类数（词汇表大小）
        search_mode: 解码模式，'max' 表示贪婪搜索，'beam' 表示集束搜索
        blank_id: CTC 空白标签的 ID
    """

    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                          num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        """执行解码。

        Args:
            nn_output: 神经网络输出 logits，形状为 (T, B, C) 或 (B, T, C)
            vid_lgt: 每个样本的序列长度
            batch_first: 输入是否以 batch 为第一维
            probs: 输入是否已经是概率值（无需 softmax）

        Returns:
            list: 解码后的词汇序列列表
        """
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """集束搜索解码。

        使用 CTCBeamDecoder 执行集束搜索解码，并去除连续重复的标签。

        Args:
            nn_output: 神经网络输出，形状为 (B, T, N)
            vid_lgt: 每个样本的序列长度
            probs: 输入是否已经是概率值

        Returns:
            list: 解码后的词汇序列列表
        """
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][:out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(first_result)])
        return ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        """贪婪搜索解码。

        取每个时间步概率最大的标签，去除连续重复和空白标签。

        Args:
            nn_output: 神经网络输出，形状为 (B, T, N)
            vid_lgt: 每个样本的序列长度

        Returns:
            list: 解码后的词汇序列列表
        """
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list