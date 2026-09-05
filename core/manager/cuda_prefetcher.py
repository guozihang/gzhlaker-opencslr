"""CUDA batch prefetching for the training and evaluation loops."""

import torch


class CUDAPrefetcher:
    """Overlap host-to-device copies with computation on the current stream."""

    def __init__(self, loader, device, gpu_augment=None):
        self.loader = loader
        self.device = torch.device(device)
        self.gpu_augment = gpu_augment
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()

    @staticmethod
    def _to_device(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        return value

    def _copy_batch(self, batch):
        video, video_length, labels, label_length, info = batch
        video = self._to_device(video, self.device)
        video_length = self._to_device(video_length, self.device)
        labels = self._to_device(labels, self.device)
        label_length = self._to_device(label_length, self.device)
        if self.gpu_augment is not None and video.ndim == 5:
            video = self.gpu_augment(video)
        return video, video_length, labels, label_length, info

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return
        iterator = iter(self.loader)
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        next_batch = None

        def preload():
            nonlocal next_batch
            try:
                batch = next(iterator)
            except StopIteration:
                next_batch = None
                return
            with torch.cuda.stream(stream):
                next_batch = self._copy_batch(batch)

        preload()
        while next_batch is not None:
            current.wait_stream(stream)
            batch = next_batch
            for value in batch[:4]:
                if isinstance(value, torch.Tensor):
                    value.record_stream(current)
            preload()
            yield batch
