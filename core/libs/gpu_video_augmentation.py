"""Batch video augmentations executed on CUDA.

The dataset keeps temporal operations in the workers because they change the
sequence length.  This module handles the shape-preserving spatial part after
the batch has been copied to the training device.
"""

import torch
import torch.nn.functional as F


class GPUVideoAugment:
    """Apply crop, flip, resize and normalization to ``B,T,C,H,W`` videos."""

    def __init__(self, crop_size=224, image_scale=1.0, training=True, flip_prob=0.5):
        self.crop_size = (int(crop_size), int(crop_size))
        self.image_scale = float(image_scale)
        self.training = bool(training)
        self.flip_prob = float(flip_prob)
        if not 0.0 <= self.flip_prob <= 1.0:
            raise ValueError("flip_prob must be between 0 and 1")

    @staticmethod
    def _pad_to_size(video, target_h, target_w):
        _, _, _, height, width = video.shape
        pad_h = max(0, target_h - height)
        pad_w = max(0, target_w - width)
        if pad_h or pad_w:
            video = F.pad(
                video,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
        return video

    def _crop(self, video):
        target_h, target_w = self.crop_size
        video = self._pad_to_size(video, target_h, target_w)
        batch, _, _, height, width = video.shape
        if self.training:
            top = torch.randint(height - target_h + 1, (batch,), device=video.device)
            left = torch.randint(width - target_w + 1, (batch,), device=video.device)
        else:
            top = video.new_full((batch,), (height - target_h) // 2, dtype=torch.long)
            left = video.new_full((batch,), (width - target_w) // 2, dtype=torch.long)

        rows = top[:, None, None, None, None] + torch.arange(
            target_h, device=video.device, dtype=torch.long
        )[None, None, None, :, None]
        rows = rows.expand(batch, video.shape[1], video.shape[2], target_h, video.shape[4])
        video = torch.gather(video, 3, rows)
        cols = left[:, None, None, None, None] + torch.arange(
            target_w, device=video.device, dtype=torch.long
        )[None, None, None, None, :]
        cols = cols.expand(batch, video.shape[1], video.shape[2], target_h, target_w)
        return torch.gather(video, 4, cols)

    def __call__(self, video):
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("GPUVideoAugment expects a B,T,C,H,W tensor")
        video = video.float()
        video = self._crop(video)
        if self.training and self.flip_prob:
            flip = torch.rand(video.shape[0], device=video.device) < self.flip_prob
            video = torch.where(flip[:, None, None, None, None], video.flip(-1), video)
        if self.image_scale != 1.0:
            height = max(1, int(round(video.shape[-2] * self.image_scale)))
            width = max(1, int(round(video.shape[-1] * self.image_scale)))
            flat = video.flatten(0, 1)
            flat = F.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
            video = flat.reshape(video.shape[0], video.shape[1], video.shape[2], height, width)
        return video / 127.5 - 1.0
