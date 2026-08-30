# -*- encoding: utf-8 -*-
"""OpenCSLR 数据收集管理器模块。

提供批次数据整理(collate)功能，负责对视频数据进行填充(padding)并以统一格式
组织批次数据，确保数据长度对齐以满足模型输入要求。
"""
from itertools import chain
import numpy as np
import torch


class CollectManager:
    """数据收集管理器。

    管理卷积核大小配置，并提供批次数据整理函数 collate_fn，
    用于 DataLoader 中对视频序列进行填充和长度对齐。
    """
    # 定义类变量用于存储卷积核大小配置
    KERNEL_SIZES = None

    @classmethod
    def init( cls, args ):
        """初始化收集管理器，设置卷积核大小配置。

        Args:
            args: 参数对象，包含 model_args.kernel_size 配置
        """
        cls.KERNEL_SIZES = args.model_args["kernel_size"]

    @staticmethod
    def collate ( batch ) :
        """整理批次数据，对视频序列进行填充和对齐。

        根据卷积核大小计算左右填充量，对视频帧序列进行填充以对齐长度，
        同时对标签也进行展平处理。支持视频数据和预提取特征两种数据类型。

        Args:
            batch: 批次数据列表，每个元素为 (video, label, info) 三元组

        Returns:
            tuple: (padded_video, video_length, padded_label, label_length, info)
                其中 padded_video 为填充后的视频张量，video_length 为各视频原始长度，
                padded_label 为展平后的标签序列，label_length 为各标签序列长度
        """
        batch = [ item for item in sorted ( batch , key = lambda x : len ( x [ 0 ] ) , reverse = True ) ]
        video , label , info = list ( zip ( *batch ) )

        left_pad = 0
        last_stride = 1
        total_stride = 1
        for layer_idx , ks in enumerate ( CollectManager.KERNEL_SIZES ) :
            if ks [ 0 ] == 'K' :
                left_pad = left_pad * last_stride
                left_pad += int ( (int ( ks [ 1 ] ) - 1) / 2 )
            elif ks [ 0 ] == 'P' :
                last_stride = int ( ks [ 1 ] )
                total_stride = total_stride * last_stride
        if len ( video [ 0 ].shape ) > 3 :
            max_len = len ( video [ 0 ] )
            video_length = torch.LongTensor (
                [ np.ceil ( len ( vid ) / total_stride ) * total_stride + 2 * left_pad for vid in video ] )
            right_pad = int ( np.ceil ( max_len / total_stride ) ) * total_stride - max_len + left_pad
            max_len = max_len + left_pad + right_pad
            padded_video = [ torch.cat (
                (
                    vid [ 0 ] [ None ].expand ( left_pad , -1 , -1 , -1 ) ,
                    vid ,
                    vid [ -1 ] [ None ].expand ( max_len - len ( vid ) - left_pad , -1 , -1 , -1 ) ,
                )
                , dim = 0 )
                for vid in video ]
            padded_video = torch.stack ( padded_video )
        else :
            max_len = len ( video [ 0 ] )
            video_length = torch.LongTensor ( [ len ( vid ) for vid in video ] )
            padded_video = [ torch.cat (
                (
                    vid ,
                    vid [ -1 ] [ None ].expand ( max_len - len ( vid ) , -1 ) ,
                )
                , dim = 0 )
                for vid in video ]
            padded_video = torch.stack ( padded_video ).permute ( 0 , 2 , 1 )
        label_length = torch.LongTensor ( [ len ( lab ) for lab in label ] )
        if max ( label_length ) == 0 :
            return padded_video , video_length , [ ] , [ ] , info
        else :
            padded_label = [ ]
            for lab in label :
                padded_label.extend ( lab )
            padded_label = torch.LongTensor ( padded_label )
            return padded_video , video_length , padded_label , label_length , info

    @classmethod
    def set_kernel_sizes(cls, kernel_sizes):
        """设置卷积核大小配置。

        Args:
            kernel_sizes: 卷积核大小列表，包含 'K'（卷积）和 'P'（池化）配置
        """
        cls.KERNEL_SIZES = kernel_sizes