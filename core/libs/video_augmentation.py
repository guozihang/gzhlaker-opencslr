"""Video data augmentation modules for CSLR training.

Provides a collection of spatial and temporal augmentation transforms
including random cropping, horizontal flipping, rotation, temporal rescaling,
and resizing. Also includes a WER-based augmentation strategy for gloss-level
perturbations.

Written by Yuecong Min
"""

import cv2
import pdb
import PIL
import copy
import scipy.misc
import torch
import random
import numbers
import numpy as np


class Compose(object):
    """Compose multiple video augmentation transforms.

    Args:
        transforms: List of transform objects to apply sequentially.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label, file_info=None):
        """Apply the composed transforms.

        WERAugment instances receive the label and file_info; other
        transforms receive only the image data.

        Args:
            image: Input video frames.
            label: Gloss labels.
            file_info: Optional file identifier for WERAugment.

        Returns:
            Tuple of (augmented_image, augmented_label).
        """
        for t in self.transforms:
            if file_info is not None and isinstance(t, WERAugment):
                image, label = t(image, label, file_info)
            else:
                image = t(image)
        return image, label


class WERAugment(object):
    """Word Error Rate (WER) based data augmentation.

    Applies random deletion, insertion, and substitution operations on
    gloss sequences to simulate recognition errors.

    Args:
        boundary_path: Path to a .npy file containing gloss boundary
            information for each video.
    """

    def __init__(self, boundary_path):
        self.boundary_dict = np.load(boundary_path, allow_pickle=True).item()
        self.K = 3

    def __call__(self, video, label, file_info):
        """Apply WER augmentation to a video sample.

        Args:
            video: List of video frames.
            label: Gloss labels.
            file_info: File identifier for boundary lookup.

        Returns:
            Tuple of (augmented_video, augmented_label).
        """
        ind = np.arange(len(video)).tolist()
        if file_info not in self.boundary_dict.keys():
            return video, label
        binfo = copy.deepcopy(self.boundary_dict[file_info])
        binfo = [0] + binfo + [len(video)]
        k = np.random.randint(min(self.K, len(label) - 1))
        for i in range(k):
            ind, label, binfo = self.one_operation(ind, label, binfo)
        ret_video = [video[i] for i in ind]
        return ret_video, label

    def one_operation(self, *inputs):
        """Randomly select one of delete, substitute, or insert operations.

        Returns:
            Tuple of (indices, label, boundary_info) after the operation.
        """
        prob = np.random.random()
        if prob < 0.3:
            return self.delete(*inputs)
        elif 0.3 <= prob < 0.7:
            return self.substitute(*inputs)
        else:
            return self.insert(*inputs)

    @staticmethod
    def delete(ind, label, binfo):
        """Delete a random gloss from the sequence.

        Args:
            ind: Frame index list.
            label: Gloss labels.
            binfo: Boundary information list.

        Returns:
            Tuple of (updated_indices, updated_label, updated_boundaries).
        """
        del_wd = np.random.randint(len(label))
        ind = ind[:binfo[del_wd]] + ind[binfo[del_wd + 1]:]
        duration = binfo[del_wd + 1] - binfo[del_wd]
        del label[del_wd]
        binfo = [i for i in binfo[:del_wd]] + [i - duration for i in binfo[del_wd + 1:]]
        return ind, label, binfo

    @staticmethod
    def insert(ind, label, binfo):
        """Insert a random gloss into the sequence.

        Args:
            ind: Frame index list.
            label: Gloss labels.
            binfo: Boundary information list.

        Returns:
            Tuple of (updated_indices, updated_label, updated_boundaries).
        """
        ins_wd = np.random.randint(len(label))
        ins_pos = np.random.choice(binfo)
        ins_lab_pos = binfo.index(ins_pos)

        ind = ind[:ins_pos] + ind[binfo[ins_wd]:binfo[ins_wd + 1]] + ind[ins_pos:]
        duration = binfo[ins_wd + 1] - binfo[ins_wd]
        label = label[:ins_lab_pos] + [label[ins_wd]] + label[ins_lab_pos:]
        binfo = binfo[:ins_lab_pos] + [binfo[ins_lab_pos - 1] + duration] + [i + duration for i in binfo[ins_lab_pos:]]
        return ind, label, binfo

    @staticmethod
    def substitute(ind, label, binfo):
        """Substitute one gloss with another in the sequence.

        Args:
            ind: Frame index list.
            label: Gloss labels.
            binfo: Boundary information list.

        Returns:
            Tuple of (updated_indices, updated_label, updated_boundaries).
        """
        sub_wd = np.random.randint(len(label))
        tar_wd = np.random.randint(len(label))

        ind = ind[:binfo[tar_wd]] + ind[binfo[sub_wd]:binfo[sub_wd + 1]] + ind[binfo[tar_wd + 1]:]
        label[tar_wd] = label[sub_wd]
        delta_duration = binfo[sub_wd + 1] - binfo[sub_wd] - (binfo[tar_wd + 1] - binfo[tar_wd])
        binfo = binfo[:tar_wd + 1] + [i + delta_duration for i in binfo[tar_wd + 1:]]
        return ind, label, binfo


class ToTensor(object):
    """Convert a video clip (list of numpy arrays) to a torch tensor.

    Transposes the array from (T, H, W, C) to (T, C, H, W) format.
    """

    def __call__(self, video):
        """Convert video to torch tensor.

        Args:
            video: List of numpy arrays or a single numpy array.

        Returns:
            Float tensor of shape (T, C, H, W).
        """
        if isinstance(video, list):
            video = np.array(video)
            video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video.transpose((0, 3, 1, 2)))
        return video


class RandomCrop(object):
    """Extract random crop of the video.

    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, clip):
        """Apply random crop to the video clip.

        Args:
            clip: List of images in numpy.ndarray or PIL.Image format.

        Returns:
            Cropped list of images.
        """
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        crop_h, crop_w = self.size
        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        if isinstance(clip[0], np.ndarray):
            return [img[h1:h1 + crop_h, w1:w1 + crop_w, :] for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.crop((w1, h1, w1 + crop_w, h1 + crop_h)) for img in clip]


class CenterCrop(object):
    """Extract center crop from the video.

    Args:
        size: Desired output size. If a single number, a square crop of that
            size is used. If a sequence, it must be of length 2 (h, w).
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clip):
        """Apply center crop to the video clip.

        Args:
            clip: List of images in numpy.ndarray format.

        Returns:
            Center-cropped list of images.
        """
        try:
            im_h, im_w, im_c = clip[0].shape
        except ValueError:
            print(clip[0].shape)
        new_h, new_w = self.size
        new_h = im_h if new_h >= im_h else new_h
        new_w = im_w if new_w >= im_w else new_w
        top = int(round((im_h - new_h) / 2.))
        left = int(round((im_w - new_w) / 2.))
        return [img[top:top + new_h, left:left + new_w] for img in clip]


class RandomHorizontalFlip(object):
    """Randomly flip video frames horizontally.

    Args:
        prob: Probability of flipping.
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, clip):
        """Apply random horizontal flip to the video clip.

        Args:
            clip: List of images in numpy.ndarray format.

        Returns:
            Flipped or original clip as a numpy array.
        """
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            clip = np.flip(clip, axis=2)
            clip = np.ascontiguousarray(copy.deepcopy(clip))
        return np.array(clip)


class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within given bounds.

    Args:
        degrees (sequence or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max),
            the range of degrees will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')
        self.degrees = degrees

    def __call__(self, clip):
        """Apply random rotation to the video clip.

        Args:
            clip: List of images in numpy.ndarray or PIL.Image format.

        Returns:
            Rotated list of images.
        """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [scipy.misc.imrotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return rotated


class TemporalRescale(object):
    """Randomly rescale the temporal length of a video clip.

    Args:
        temp_scaling: Scaling factor range (1 - temp_scaling, 1 + temp_scaling).
        frame_interval: Frame sampling interval used to compute max length.
    """

    def __init__(self, temp_scaling=0.2, frame_interval=1):
        self.min_len = 32
        self.max_len = int(np.ceil(230 / frame_interval))
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, clip):
        """Apply temporal rescaling to the video clip.

        Args:
            clip: List of video frames.

        Returns:
            Temporally rescaled list of frames.
        """
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        return clip[index]


class RandomResize(object):
    """Resize video by zooming in and out with a random scaling factor.

    Args:
        rate (float): Video is scaled uniformly between [1 - rate, 1 + rate].
        interp (string): Interpolation to use for re-sizing
            ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        """Apply random resize to the video clip.

        Args:
            clip: List of images in numpy.ndarray or PIL.Image format.

        Returns:
            Resized list of images.
        """
        scaling_factor = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_h, new_w)
        if isinstance(clip[0], np.ndarray):
            return [scipy.misc.imresize(img, size=(new_h, new_w), interp=self.interpolation) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        """Map interpolation string to PIL.Image constant.

        Args:
            interp: Interpolation method name.

        Returns:
            PIL.Image interpolation constant.
        """
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC


class Resize(object):
    """Resize video by a fixed scaling factor.

    Args:
        rate (float): Fixed scaling factor for resizing.
        interp (string): Interpolation to use for re-sizing
            ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate
        self.interpolation = interp

    def __call__(self, clip):
        """Apply fixed resize to the video clip.

        Args:
            clip: List of images in numpy.ndarray or PIL.Image format.

        Returns:
            Resized list of images.
        """
        if self.rate == 1.0:
            return clip
        scaling_factor = self.rate

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        if isinstance(clip[0], np.ndarray):
            return [np.array(PIL.Image.fromarray(img).resize(new_size)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.resize(size=(new_w, new_h), resample=self._get_PIL_interp(self.interpolation)) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

    def _get_PIL_interp(self, interp):
        """Map interpolation string to PIL.Image constant.

        Args:
            interp: Interpolation method name.

        Returns:
            PIL.Image interpolation constant.
        """
        if interp == 'nearest':
            return PIL.Image.NEAREST
        elif interp == 'lanczos':
            return PIL.Image.LANCZOS
        elif interp == 'bilinear':
            return PIL.Image.BILINEAR
        elif interp == 'bicubic':
            return PIL.Image.BICUBIC
        elif interp == 'cubic':
            return PIL.Image.CUBIC
