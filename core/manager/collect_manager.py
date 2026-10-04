# -*- encoding: utf-8 -*-
from itertools import chain
import numpy as np
import torch


class CollectManager:
    # 定义类变量用于存储卷积核大小配置
    KERNEL_SIZES = None

    @classmethod
    def init( cls, args ):
        cls.KERNEL_SIZES = args.model_args["kernel_size"]

    @staticmethod
    def collate ( batch ) :
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
        cls.KERNEL_SIZES = kernel_sizes