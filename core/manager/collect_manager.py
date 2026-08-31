# -*- encoding: utf-8 -*-
"""OpenCSLR 数据收集管理器模块。

提供 DataLoader 的批次整理功能，负责对变长视频或时序特征进行 padding，
并生成模型所需的长度与标签张量。
"""
import numpy as np
import torch


class CollectManager:
    """管理批次整理配置并提供兼容的 ``collate`` 入口。"""

    KERNEL_SIZES = None

    @classmethod
    def init(cls, args):
        """从参数对象初始化时序层配置。"""
        try:
            kernel_sizes = args.model_args["kernel_size"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError("model_args.kernel_size is required for batch collation") from exc
        cls.set_kernel_sizes(kernel_sizes)

    @classmethod
    def _normalize_kernel_sizes(cls, kernel_sizes):
        """校验并复制 ``Kx``/``Px`` 格式的时序层配置。"""
        if kernel_sizes is None:
            raise ValueError("kernel_size must be configured before collating a batch")
        normalized = []
        for index, spec in enumerate(kernel_sizes):
            if not isinstance(spec, str) or len(spec) < 2 or spec[0] not in ("K", "P"):
                raise ValueError(
                    f"kernel_size[{index}] must use 'K<number>' or 'P<number>', got {spec!r}"
                )
            try:
                size = int(spec[1:])
            except ValueError as exc:
                raise ValueError(f"kernel_size[{index}] has invalid size: {spec!r}") from exc
            if size <= 0:
                raise ValueError(f"kernel_size[{index}] must be positive, got {spec!r}")
            normalized.append(f"{spec[0]}{size}")
        return normalized

    @classmethod
    def _padding_params(cls):
        """计算 raw video 所需的左 padding 和总 stride。"""
        left_pad = 0
        last_stride = 1
        total_stride = 1
        for spec in cls._normalize_kernel_sizes(cls.KERNEL_SIZES):
            operator, size = spec[0], int(spec[1:])
            if operator == "K":
                left_pad = left_pad * last_stride + (size - 1) // 2
            else:
                last_stride = size
                total_stride *= size
        return left_pad, total_stride

    @staticmethod
    def _as_tensor(value, name):
        """将 numpy 输入转换为 Tensor，并提供统一错误信息。"""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value)
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray, got {type(value).__name__}")

    @classmethod
    def _collate_video(cls, videos):
        """整理 ``(T,C,H,W)`` raw video，并复制边界帧完成 padding。"""
        left_pad, total_stride = cls._padding_params()
        max_len = len(videos[0])
        video_length = torch.LongTensor([
            int(np.ceil(len(video) / total_stride) * total_stride + 2 * left_pad)
            for video in videos
        ])
        right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
        padded_len = max_len + left_pad + right_pad
        padded = []
        for video in videos:
            padded.append(torch.cat((
                video[0][None].expand(left_pad, -1, -1, -1),
                video,
                video[-1][None].expand(padded_len - len(video) - left_pad, -1, -1, -1),
            ), dim=0))
        return torch.stack(padded), video_length

    @staticmethod
    def _collate_features(videos):
        """整理 ``(T,D)`` 时序特征，不应用 raw video 的时序 padding 规则。"""
        max_len = len(videos[0])
        video_length = torch.LongTensor([len(video) for video in videos])
        padded = [torch.cat((
            video,
            video[-1][None].expand(max_len - len(video), -1),
        ), dim=0) for video in videos]
        return torch.stack(padded).permute(0, 2, 1), video_length

    @staticmethod
    def _collate_labels(labels):
        """展平标签，并始终返回稳定的 LongTensor 类型。"""
        label_length = torch.LongTensor([len(label) for label in labels])
        label_tensors = [torch.as_tensor(label, dtype=torch.long).reshape(-1) for label in labels]
        padded_label = torch.cat(label_tensors) if label_tensors else torch.empty(0, dtype=torch.long)
        return padded_label, label_length

    @classmethod
    def collate(cls, batch):
        """整理一个变长 batch，返回兼容的五元组。

        Returns:
            tuple: ``(padded_video, video_length, padded_label, label_length, info)``。
                raw video 输出为 ``(B,T,C,H,W)``，特征输出为 ``(B,C,T)``；
                ``video_length`` 表示模型时间轴上的有效长度。
        """
        if not batch:
            raise ValueError("cannot collate an empty batch")
        if any(not isinstance(item, (tuple, list)) or len(item) != 3 for item in batch):
            raise ValueError("each batch item must be a (video, label, info) tuple")

        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)
        videos, labels, info = zip(*batch)
        videos = [cls._as_tensor(video, "video") for video in videos]
        if any(video.ndim != videos[0].ndim for video in videos):
            raise ValueError("all videos in a batch must have the same number of dimensions")
        if any(len(video) == 0 for video in videos):
            raise ValueError("videos must contain at least one time step")
        if videos[0].ndim == 4:
            padded_video, video_length = cls._collate_video(videos)
        elif videos[0].ndim == 2:
            padded_video, video_length = cls._collate_features(videos)
        else:
            raise ValueError(
                f"unsupported video shape {tuple(videos[0].shape)}; expected (T,C,H,W) or (T,D)"
            )
        padded_label, label_length = cls._collate_labels(labels)
        return padded_video, video_length, padded_label, label_length, info

    @classmethod
    def set_kernel_sizes(cls, kernel_sizes):
        """设置并复制 ``Kx``/``Px`` 格式的卷积核配置。"""
        cls.KERNEL_SIZES = cls._normalize_kernel_sizes(kernel_sizes)
