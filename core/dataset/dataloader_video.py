"""Video dataset loader for CSLR.

Supports multiple data types: video (raw jpg), lmdb, memmap, and pre-extracted
features. Returns video frames, label indices, and original file info for each
sample.
"""
import os
import cv2
import sys
import glob
import torch
import numpy as np
import torch.utils.data as data
from libs import video_augmentation
import pickle
sys.path.append("..")

_MEMMAP_CACHE = {}
_INPUTS_CACHE = {}


class VideoDataset(data.Dataset):
    """Video dataset for continuous sign language recognition.

    Loads video frames and corresponding gloss labels from different storage
    backends (video folders, LMDB, memmap, or pre-extracted features). Applies
    data augmentation during training mode.

    Args:
        prefix: Root path prefix for the dataset.
        gloss_dict: Dictionary mapping gloss strings to indices.
        dataset: Dataset name (e.g., 'phoenix2014', 'phoenix2014-T', 'CSL-Daily').
        drop_ratio: Ratio for dropping frames (unused).
        num_gloss: Number of glosses in the dictionary.
        mode: Dataset split, one of 'train', 'dev', 'test'.
        transform_mode: Whether to apply training transforms.
        datatype: Data storage type ('video', 'lmdb', 'memmap', or other for features).
        frame_interval: Frame sampling interval.
        image_scale: Scale factor for image resize.
        allowable_vid_length: Minimum allowable video length.
        skip_fileids: File IDs to skip (string, list, or dict per mode).
        skip_indices: Sample indices to skip (string, list, or dict per mode).
        skip_info_path: Path to a file listing file IDs to skip.
        preprocess_root: Root directory holding '{dataset}/{mode}_info.npy' files.
        memmap_root: Root directory holding the per-dataset memmap bigarray
            folders (required when datatype is 'memmap').
        feature_root: Root directory holding '{mode}/{fileid}_features.npy'
            pre-extracted feature files.
        image_feature_dir: Directory of full-frame images, relative to prefix.
        memmap_layout: Dict describing the per-dataset memmap layout
            (subdir/pickle_name/memmap_name, name templates may contain
            '{mode}'), provided by configs/dataset.yaml via dataset_info.
        memmap_frame_shape: Shape of a single frame in the memmap bigarray,
            provided by configs/dataset.yaml.
        limit_len: If set, cap the dataset length to this many samples
            (for quick pipeline smoke tests).
    """

    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, allowable_vid_length=16,
                 skip_fileids=None, skip_indices=None, skip_info_path=None,
                 preprocess_root="./preprocess", memmap_root=None, feature_root="./features",
                 image_feature_dir="features/fullFrame-256x256px", limit_len=None,
                 memmap_layout=None, memmap_frame_shape=(256, 256, 3),
                 cache_file_lists=True, cache_features=True, precache_file_lists=False,
                 gpu_augment=False):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.allowable_vid_length = allowable_vid_length
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.preprocess_root = preprocess_root
        self.memmap_root = memmap_root
        self.feature_root = feature_root
        self.image_feature_dir = image_feature_dir
        self.memmap_layout = memmap_layout
        self.memmap_frame_shape = tuple(memmap_frame_shape) if memmap_frame_shape is not None else None
        self.feat_prefix = os.path.join(prefix, image_feature_dir, mode)
        self.limit_len = limit_len
        self.cache_file_lists = bool(cache_file_lists)
        self.cache_features = bool(cache_features)
        # GPU augmentation is executed by CUDAPrefetcher after collation.  The
        # dataset only changes its CPU transform pipeline here.
        self.gpu_augment = bool(gpu_augment and torch.cuda.is_available())
        self._file_list_cache = {}
        self._feature_cache = {}
        self._label_cache = {}
        self.transform_mode = "train" if transform_mode else "test"
        skip_fileids = self.select_mode_value(skip_fileids, mode)
        skip_indices = self.select_mode_value(skip_indices, mode)
        skip_info_path = self.select_mode_value(skip_info_path, mode)
        self.inputs_list, self.input_indices = self.load_inputs_list(
            dataset, mode, preprocess_root, skip_fileids, skip_indices, skip_info_path
        )
        print(mode, len(self))
        self.data_aug = self.transform()
        if self.data_type == "video" and precache_file_lists:
            self._precache_file_lists()

    def _video_glob(self, fi):
        base_folder = os.path.join(self.prefix, self.image_feature_dir, fi['folder'])
        return base_folder if 'phoenix' in self.dataset else os.path.join(base_folder, "*.jpg")

    def _precache_file_lists(self):
        for fi in self.inputs_list:
            pattern = self._video_glob(fi)
            if pattern not in self._file_list_cache:
                self._file_list_cache[pattern] = sorted(glob.glob(pattern))

    def _labels(self, fi):
        fileid = fi.get("fileid", fi.get("original_info", ""))
        if fileid not in self._label_cache:
            self._label_cache[fileid] = [
                self.dict[phase][0]
                for phase in fi.get("label", "").split()
                if phase in self.dict
            ]
        return list(self._label_cache[fileid])

    @staticmethod
    def select_mode_value(value, mode):
        """Select value based on mode if value is a dict.

        Args:
            value: Raw value, possibly a dict with mode keys.
            mode: Current mode string ('train', 'dev', 'test').

        Returns:
            The value for the given mode, the 'all' key, or the raw value.
        """
        if isinstance(value, dict):
            return value.get(mode, value.get("all"))
        return value

    @staticmethod
    def parse_list(value):
        """Parse a value into a list of strings.

        Args:
            value: None, a list/tuple/set, or a string with comma or newline separators.

        Returns:
            A list of strings.
        """
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, str):
            value = value.replace(",", "\n")
            return [item.strip() for item in value.splitlines() if item.strip()]
        return [value]

    @staticmethod
    def normalize_fileid(value):
        """Normalize a file ID string by stripping path and extension.

        Args:
            value: Raw file ID string (may contain path or '.npy' suffix).

        Returns:
            Normalized file ID (basename without extension).
        """
        value = str(value).strip()
        value = value.split("|")[0]
        value = os.path.basename(value)
        if value.endswith(".npy"):
            value = value[:-4]
        return value

    @staticmethod
    def is_sample_item(index, item):
        """Check if an item is a valid sample entry.

        Args:
            index: Sample index.
            item: Data item to check.

        Returns:
            True if the index is numeric and the item is a dict with 'fileid'.
        """
        try:
            int(index)
        except (TypeError, ValueError):
            return False
        return isinstance(item, dict) and "fileid" in item

    @classmethod
    def load_inputs_list(cls, dataset, mode, preprocess_root="./preprocess",
                         skip_fileids=None, skip_indices=None, skip_info_path=None):
        """Load and filter the list of input samples.

        Args:
            dataset: Dataset name.
            mode: Dataset split ('train', 'dev', 'test').
            preprocess_root: Root directory holding per-dataset info files.
            skip_fileids: File IDs to skip.
            skip_indices: Sample indices to skip.
            skip_info_path: Path to a file with file IDs to skip.

        Returns:
            Tuple of (filtered_inputs_list, filtered_indices_list).
        """
        info_path = os.path.join(preprocess_root, dataset, f"{mode}_info.npy")
        cache_key = os.path.abspath(info_path)
        indexed_inputs = _INPUTS_CACHE.get(cache_key)
        if indexed_inputs is None:
            raw_inputs = np.load(info_path, allow_pickle=True).item()
            if isinstance(raw_inputs, dict):
                indexed_inputs = sorted(
                    [(int(idx), item) for idx, item in raw_inputs.items() if cls.is_sample_item(idx, item)],
                    key=lambda item: item[0]
                )
            else:
                indexed_inputs = list(enumerate(raw_inputs))
            _INPUTS_CACHE[cache_key] = indexed_inputs

        skipped_fileids = {cls.normalize_fileid(item) for item in cls.parse_list(skip_fileids)}
        if skip_info_path:
            with open(skip_info_path, "r", encoding="utf-8") as reader:
                skipped_fileids.update(
                    cls.normalize_fileid(line)
                    for line in reader
                    if line.strip() and not line.lstrip().startswith("#")
                )
        skipped_indices = {int(item) for item in cls.parse_list(skip_indices)}

        if not skipped_fileids and not skipped_indices:
            return [item for _, item in indexed_inputs], [idx for idx, _ in indexed_inputs]

        kept_inputs = []
        kept_indices = []
        for original_idx, item in indexed_inputs:
            fileid = cls.normalize_fileid(item.get("fileid", ""))
            original_info = cls.normalize_fileid(item.get("original_info", ""))
            if int(original_idx) in skipped_indices or fileid in skipped_fileids or original_info in skipped_fileids:
                continue
            kept_inputs.append(item)
            kept_indices.append(original_idx)

        print(f"Skip {len(indexed_inputs) - len(kept_inputs)} samples for {dataset}/{mode}.")
        return kept_inputs, kept_indices

    def __getitem__(self, idx):
        """Get a sample by index.

        Dispatches to the appropriate reader based on data_type and applies
        normalization / augmentation.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (video_tensor, label_tensor, original_info_string).
        """
        if self.data_type == "video":
            input_data, label, _ = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, _ = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "memmap":
            if not hasattr(self, 'mem'):
                self.init_memmap ( )
            input_data , label , fi = self.read_memmap ( idx )
            input_data , label = self.normalize ( input_data , label )
            return input_data , torch.LongTensor ( label ) , self.inputs_list [ idx ] [ 'original_info' ]
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def init_memmap(self):
        """Initialize memory-mapped numpy arrays for video data.

        Loads pre-computed pickle info files and opens the corresponding
        memmap files for the current dataset and mode. File locations are
        derived from ``memmap_root`` and the per-dataset ``memmap_layout``
        (both provided by configs/dataset.yaml via dataset_info).

        Raises:
            ValueError: If ``memmap_root`` / ``memmap_layout`` /
                ``memmap_frame_shape`` is not configured.
        """
        cache_key = (
            self.dataset, self.mode, os.path.abspath(self.memmap_root or ""),
            repr(self.memmap_layout), self.memmap_frame_shape,
        )
        cached = _MEMMAP_CACHE.get(cache_key)
        if cached is not None:
            self.info, self.mem = cached
            return
        if self.memmap_root is None:
            raise ValueError(
                f"'memmap_root' is not configured for dataset '{self.dataset}'. "
                f"Set memmap_root in configs/dataset.yaml."
            )
        if self.memmap_layout is None:
            raise ValueError(
                f"'memmap_layout' is not configured for dataset '{self.dataset}'. "
                f"Set memmap_layout in configs/dataset.yaml."
            )
        if self.memmap_frame_shape is None:
            raise ValueError(
                f"'memmap_frame_shape' is not configured for dataset '{self.dataset}'. "
                f"Set memmap_frame_shape in configs/dataset.yaml."
            )
        layout = self.memmap_layout
        subdir, pickle_name, memmap_name = layout["subdir"], layout["pickle_name"], layout["memmap_name"]
        layout_dir = os.path.join(self.memmap_root, subdir)
        pickle_path = os.path.join(layout_dir, pickle_name.format(mode=self.mode))
        memmap_path = os.path.join(layout_dir, memmap_name.format(mode=self.mode))
        with open(pickle_path, mode="rb") as f:
            self.info = pickle.load(f)
        T = self.info[-1]['end']
        self.info = {i["path"].split("/")[-1]: [i["start"], i["end"]] for i in self.info}
        self.mem = np.memmap(memmap_path, mode="r", shape=(T, *self.memmap_frame_shape))
        _MEMMAP_CACHE[cache_key] = (self.info, self.mem)

    def read_memmap(self, index):
        """Read a sample from a memory-mapped array.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_list, label_list, file_info_dict).
        """
        fi = self.inputs_list [ index ]
        label_list = self._labels(fi)
        images = self.mem [ self.info [ fi [ 'fileid' ] + ".npy" ] [ 0 ] :self.info [ fi [ 'fileid' ] + ".npy" ] [ 1 ] ]
        return images , label_list , fi

    def read_video(self, index):
        """Read a sample from on-disk JPEG image folders.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_list, label_list, file_info_dict).
        """
        fi = self.inputs_list[index]
        img_folder = self._video_glob(fi)
        if self.cache_file_lists:
            img_list = self._file_list_cache.get(img_folder)
            if img_list is None:
                img_list = sorted(glob.glob(img_folder))
                self._file_list_cache[img_folder] = img_list
        else:
            img_list = sorted(glob.glob(img_folder))

        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = self._labels(fi)
        frames = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]
        if not frames:
            raise RuntimeError(f"No frames found for video folder: {img_folder}")
        return np.stack(frames, axis=0), label_list, fi

    def read_features(self, index):
        """Read a sample from pre-extracted feature files.

        Args:
            index: Sample index.

        Returns:
            Tuple of (features, labels).
        """
        fi = self.inputs_list[index]
        feature_path = os.path.join(self.feature_root, self.mode, f"{fi['fileid']}_features.npy")
        data = self._feature_cache.get(feature_path) if self.cache_features else None
        if data is None:
            data = np.load(feature_path, allow_pickle=True).item()
            if self.cache_features:
                self._feature_cache[feature_path] = data
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        """Apply data augmentation and normalize pixel values to [-1, 1].

        Args:
            video: Input video frames.
            label: Gloss label indices.
            file_id: Optional file identifier for WERAugment.

        Returns:
            Tuple of (normalized_video_tensor, label).
        """
        video, label = self.data_aug(video, label, file_id)
        if not self.gpu_augment:
            video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        """Build the data augmentation pipeline.

        Returns:
            A Compose object with the appropriate transforms for train or test mode.
        """
        if self.transform_mode == "train":
            print("Apply training transform.")
            if self.gpu_augment:
                return video_augmentation.Compose([
                    video_augmentation.ToTensor(),
                    video_augmentation.TemporalRescale(0.2, self.frame_interval),
                ])
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            if self.gpu_augment:
                return video_augmentation.Compose([
                    video_augmentation.ToTensor(),
                ])
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def __len__(self):
        """Return the total number of samples.

        Returns:
            Number of samples in the dataset (capped by limit_len if set).
        """
        if self.limit_len is not None:
            return min(len(self.inputs_list), self.limit_len)
        return len(self.inputs_list)

    def sample_lengths(self):
        """Return approximate raw sequence lengths for optional bucketing."""
        if self.data_type == "memmap":
            self.init_memmap()
            return [
                self.info[fi["fileid"] + ".npy"][1] - self.info[fi["fileid"] + ".npy"][0]
                for fi in self.inputs_list[:len(self)]
            ]
        if self.data_type == "video":
            if not self.cache_file_lists:
                return [1] * len(self)
            self._precache_file_lists()
            return [len(self._file_list_cache[self._video_glob(fi)]) for fi in self.inputs_list[:len(self)]]
        return [1] * len(self)

    def __getstate__(self):
        """Avoid serializing an open memmap when workers use spawn."""
        state = self.__dict__.copy()
        state.pop("mem", None)
        return state
