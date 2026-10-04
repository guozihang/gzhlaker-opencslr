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

class VideoDataset( data.Dataset ):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, allowable_vid_length=16,
                 skip_fileids=None, skip_indices=None, skip_info_path=None):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.allowable_vid_length = allowable_vid_length
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        skip_fileids = self.select_mode_value(skip_fileids, mode)
        skip_indices = self.select_mode_value(skip_indices, mode)
        skip_info_path = self.select_mode_value(skip_info_path, mode)
        self.inputs_list, self.input_indices = self.load_inputs_list(
            dataset, mode, skip_fileids, skip_indices, skip_info_path
        )
        print(mode, len(self))
        self.data_aug = self.transform()

    @staticmethod
    def select_mode_value(value, mode):
        if isinstance(value, dict):
            return value.get(mode, value.get("all"))
        return value

    @staticmethod
    def parse_list(value):
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
        value = str(value).strip()
        value = value.split("|")[0]
        value = os.path.basename(value)
        if value.endswith(".npy"):
            value = value[:-4]
        return value

    @staticmethod
    def is_sample_item(index, item):
        try:
            int(index)
        except (TypeError, ValueError):
            return False
        return isinstance(item, dict) and "fileid" in item

    @classmethod
    def load_inputs_list(cls, dataset, mode, skip_fileids=None, skip_indices=None, skip_info_path=None):
        raw_inputs = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
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

    def init_memmap ( self ) :
        if self.dataset == 'phoenix2014' or self.dataset == 'phoenix2014-normal':
            with open ( f"/sda/data/guozihang/sign_dataset/ph_bigarray/phoenix2014-{self.mode}.pickle" , mode = "rb" ) as f :
                self.info = pickle.load ( f )
            T = self.info [ -1 ] [ 'end' ]
            self.info = {i [ "path" ].split ( "/" ) [ -1 ] : [ i [ "start" ] , i [ "end" ] ] for i in self.info}
            self.mem = np.memmap (
                f"/sda/data/guozihang/sign_dataset/ph_bigarray/phoenix2014-bigarray-map-{self.mode}" ,
                mode = "r" ,
                shape = (T , 256 , 256 , 3) )
        elif self.dataset == 'phoenix2014-T' :
            with open (
                    f"/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-{self.mode}.pickle" ,
                    mode = "rb" ) as f :
                self.info = pickle.load ( f )
            T = self.info [ -1 ] [ 'end' ]
            self.info = {i [ "path" ].split ( "/" ) [ -1 ] : [ i [ "start" ] , i [ "end" ] ] for i in self.info}
            self.mem = np.memmap (
                f"/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-bigarray-map-{self.mode}" ,
                mode = "r" ,
                shape = (T , 256 , 256 , 3) )
        elif self.dataset == "CSL-Daily" :
            with open ( f"/sda/data/guozihang/sign_dataset/csldaily_bigarray/csldaily.pickle" , mode = "rb" ) as f :
                self.info = pickle.load ( f )
            T = self.info [ -1 ] [ 'end' ]
            self.info = {i [ "path" ].split ( "/" ) [ -1 ] : [ i [ "start" ] , i [ "end" ] ] for i in self.info}
            self.mem = np.memmap (
                f"/sda/data/guozihang/sign_dataset/csldaily_bigarray/csldaily" ,
                mode = "r" ,
                shape = (T , 256 , 256 , 3) )

    def read_memmap ( self , index ) :
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
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder']) if 'phoenix' in self.dataset else os.path.join(self.prefix, "features/fullFrame-256x256px/" + fi['folder'] + "/*.jpg")
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
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
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
        #return 100
        return len(self.inputs_list)
