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
    """

    # Per-dataset memmap layout: dataset -> (sub-directory under memmap_root,
    # info pickle file name template, memmap file name template). Templates may
    # contain a '{mode}' placeholder.
    MEMMAP_LAYOUT = {
        "phoenix2014": ("ph_bigarray", "phoenix2014-{mode}.pickle", "phoenix2014-bigarray-map-{mode}"),
        "phoenix2014-normal": ("ph_bigarray", "phoenix2014-{mode}.pickle", "phoenix2014-bigarray-map-{mode}"),
        "phoenix2014-T": ("pht_bigarray", "phoenix2014-T-{mode}.pickle", "phoenix2014-T-bigarray-map-{mode}"),
        "CSL-Daily": ("csldaily_bigarray", "csldaily.pickle", "csldaily"),
    }
    MEMMAP_FRAME_SHAPE = (256, 256, 3)

    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, allowable_vid_length=16,
                 skip_fileids=None, skip_indices=None, skip_info_path=None,
                 preprocess_root="./preprocess", memmap_root=None, feature_root="./features",
                 image_feature_dir="features/fullFrame-256x256px"):
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
        self.feat_prefix = os.path.join(prefix, image_feature_dir, mode)
        self.transform_mode = "train" if transform_mode else "test"
        skip_fileids = self.select_mode_value(skip_fileids, mode)
        skip_indices = self.select_mode_value(skip_indices, mode)
        skip_info_path = self.select_mode_value(skip_info_path, mode)
        self.inputs_list, self.input_indices = self.load_inputs_list(
            dataset, mode, preprocess_root, skip_fileids, skip_indices, skip_info_path
        )
        print(mode, len(self))
        self.data_aug = self.transform()

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
        raw_inputs = np.load(info_path, allow_pickle=True).item()
        if isinstance(raw_inputs, dict):
            indexed_inputs = sorted(
                [(int(idx), item) for idx, item in raw_inputs.items() if cls.is_sample_item(idx, item)],
                key=lambda item: item[0]
            )
        else:
            indexed_inputs = list(enumerate(raw_inputs))

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
            if hasattr ( self , 'mem' ) == False :
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
        derived from ``memmap_root`` and the per-dataset ``MEMMAP_LAYOUT``.

        Raises:
            ValueError: If ``memmap_root`` is not configured or the dataset
                has no known memmap layout.
        """
        if self.memmap_root is None:
            raise ValueError(
                f"'memmap_root' is not configured for dataset '{self.dataset}'. "
                f"Set 'memmap_root' in configs/{self.dataset}.yaml."
            )
        if self.dataset not in self.MEMMAP_LAYOUT:
            raise ValueError(
                f"No memmap layout registered for dataset '{self.dataset}'. "
                f"Known datasets: {sorted(self.MEMMAP_LAYOUT)}."
            )
        subdir, pickle_name, memmap_name = self.MEMMAP_LAYOUT[self.dataset]
        layout_dir = os.path.join(self.memmap_root, subdir)
        pickle_path = os.path.join(layout_dir, pickle_name.format(mode=self.mode))
        memmap_path = os.path.join(layout_dir, memmap_name.format(mode=self.mode))
        with open(pickle_path, mode="rb") as f:
            self.info = pickle.load(f)
        T = self.info[-1]['end']
        self.info = {i["path"].split("/")[-1]: [i["start"], i["end"]] for i in self.info}
        self.mem = np.memmap(memmap_path, mode="r", shape=(T, *self.MEMMAP_FRAME_SHAPE))

    def read_memmap(self, index):
        """Read a sample from a memory-mapped array.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_list, label_list, file_info_dict).
        """
        fi = self.inputs_list [ index ]
        label_list = [ ]
        for phase in fi [ 'label' ].split ( " " ) :
            if phase == '' :
                continue
            if phase in self.dict.keys ( ) :
                label_list.append ( self.dict [ phase ] [ 0 ] )
        images = self.mem [ self.info [ fi [ 'fileid' ] + ".npy" ] [ 0 ] :self.info [ fi [ 'fileid' ] + ".npy" ] [ 1 ] ]
        images = np.split ( images , images.shape [ 0 ] , axis = 0 )
        images = [ np.squeeze ( im , axis = 0 ) for im in images ]
        return images , label_list , fi

    def read_video(self, index):
        """Read a sample from on-disk JPEG image folders.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_list, label_list, file_info_dict).
        """
        fi = self.inputs_list[index]
        base_folder = os.path.join(self.prefix, self.image_feature_dir, fi['folder'])
        img_folder = base_folder if 'phoenix' in self.dataset else os.path.join(base_folder, "*.jpg")
        img_list = sorted(glob.glob(img_folder))

        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        """Read a sample from pre-extracted feature files.

        Args:
            index: Sample index.

        Returns:
            Tuple of (features, labels).
        """
        fi = self.inputs_list[index]
        feature_path = os.path.join(self.feature_root, self.mode, f"{fi['fileid']}_features.npy")
        data = np.load(feature_path, allow_pickle=True).item()
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
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        """Build the data augmentation pipeline.

        Returns:
            A Compose object with the appropriate transforms for train or test mode.
        """
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def __len__(self):
        """Return the total number of samples.

        Returns:
            Number of samples in the dataset.
        """
        #return 100
        return len(self.inputs_list)
