"""Unified data preprocessing for Phoenix2014, Phoenix2014-T and CSL-Daily.

Merged from:
    dataset_preprocess.py             (Phoenix2014)
    dataset_preprocess-T.py           (Phoenix2014-T)
    dataset_preprocess-CSL-Daily.py   (CSL-Daily)

Usage:
    python dataset_preprocess.py --dataset phoenix2014  --dataset-root /path/to/phoenix-2014-multisigner --process-image
    python dataset_preprocess.py --dataset phoenix2014-T --dataset-root /path/to/PHOENIX-2014-T --process-image
    python dataset_preprocess.py --dataset CSL-Daily     --dataset-root /path/to/frames --target-path /path/to/target --process-image
"""
import re
import os
import cv2
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px', check_broken=False):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    if img is None and check_broken:
        print(f'image destroyed: {img_path}, please manually modify the numframes')
        return None
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


# ---------------------------------------------------------------------------
# Phoenix2014 / Phoenix2014-T
# ---------------------------------------------------------------------------

def phoenix2014_csv2dict(anno_path, dataset_type):
    inputs_list = pandas.read_csv(anno_path)
    if dataset_type == 'train':
        broken_data = [2390]
        inputs_list.drop(broken_data, inplace=True)
    inputs_list = (inputs_list.to_dict()['id|folder|signer|annotation'].values())
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/features/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        fileid, folder, signer, label = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{folder}"))
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': f"{dataset_type}/{folder}",
            'signer': signer,
            'label': label,
            'num_frames': num_frames,
            'original_info': file_info,
        }
    return info_dict


def phoenix2014_t_csv2dict(anno_path, dataset_type):
    inputs_list = pandas.read_csv(anno_path)
    inputs_list = (inputs_list.to_dict()['name|video|start|end|speaker|orth|translation'].values())
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/features/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        name, video, start, end, speaker, orth, translation = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{video}"))
        info_dict[file_idx] = {
            'fileid': name,
            'folder': f"{dataset_type}/{video}",
            'signer': speaker,
            'label': orth,
            'num_frames': num_frames,
            'original_info': file_info,
        }
    return info_dict


def phoenix_resize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        os.makedirs(rs_img_dir, exist_ok=True)
        cv2.imwrite(rs_img_path, rs_img)


def process_phoenix(args, csv2dict):
    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.dataset_root}/{args.annotation_prefix.format(md)}", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groundtruth stm for evaluation
        generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
        # resize images
        video_index = np.arange(len(information) - 1)  # -1: info_dict holds a 'prefix' entry
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            if args.multiprocessing:
                run_mp_cmd(10, partial(phoenix_resize_dataset, dsize=args.output_res, info_dict=information), video_index)
            else:
                for idx in tqdm(video_index):
                    run_cmd(partial(phoenix_resize_dataset, dsize=args.output_res, info_dict=information), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)


# ---------------------------------------------------------------------------
# CSL-Daily
# ---------------------------------------------------------------------------

def csl_daily_csv2dict(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        inputs_list = f.readlines()
    info_dict = dict()
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list[1:]), total=len(inputs_list) - 1):  # Exclude first line
        index, name, length, gloss, char, word, postag = file_info.strip().split("|")
        info_dict[file_idx] = {
            'fileid': name,
            'folder': name + '/*.jpg',
            'signer': 'unknown',
            'label': gloss,
            'num_frames': int(length),
            'original_info': "|".join(file_info.split("|")[1:]),  # start from 'name', for model inference
        }
    return info_dict


def csl_daily_resize_dataset(video_idx, dsize, info_dict, dataset_root, target_path):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{dataset_root}/{info['folder']}")
    if len(img_list) == len(glob.glob(f"{target_path}/{info['folder']}")):
        return
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize, check_broken=True)
        if rs_img is None:
            info_dict[video_idx]['num_frames'] -= 1
            continue
        rs_img_path = f"{target_path}/{info['fileid']}/{img_path.split('/')[-1]}"
        rs_img_dir = os.path.dirname(rs_img_path)
        os.makedirs(rs_img_dir, exist_ok=True)
        cv2.imwrite(rs_img_path, rs_img)


def process_csl_daily(args):
    mode = ["train", "dev", "test"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")

    # generate information dict
    information = csl_daily_csv2dict(f"./{args.dataset}/{args.annotation_file}")
    video_index = np.arange(len(information))
    # resize images
    print(f"Resize image to {args.output_res}")
    if args.process_image:
        if args.multiprocessing:
            run_mp_cmd(100, partial(csl_daily_resize_dataset, dsize=args.output_res, info_dict=information,
                                    dataset_root=args.dataset_root, target_path=args.target_path), video_index)
        else:
            for idx in tqdm(video_index):
                run_cmd(partial(csl_daily_resize_dataset, dsize=args.output_res, info_dict=information,
                                dataset_root=args.dataset_root, target_path=args.target_path), idx)
    else:
        print("Don't resize images")

    # pack information per split
    with open(f"./{args.dataset}/{args.split_file}", 'r', encoding='utf-8') as f:
        files_list = f.readlines()
    train_files = []
    dev_files = []
    test_files = []
    for file_idx, file_info in tqdm(enumerate(files_list[1:]), total=len(files_list) - 1):  # Exclude first line
        name, split = file_info.strip().split("|")
        if split == 'train':
            train_files.append(name)
        elif split == 'dev':
            dev_files.append(name)
        elif split == 'test':
            test_files.append(name)
    assert len(train_files) + len(dev_files) + len(test_files) == len(information)
    information_pack = dict()
    for md in mode:
        information_pack[md] = dict()
    train_id = 0
    dev_id = 0
    test_id = 0
    for info_key, info_data in information.items():
        if info_data['fileid'] in train_files:
            information_pack['train'][train_id] = info_data
            train_id += 1
        elif info_data['fileid'] in dev_files:
            information_pack['dev'][dev_id] = info_data
            dev_id += 1
        elif info_data['fileid'] in test_files:
            information_pack['test'][test_id] = info_data
            test_id += 1
        else:
            information_pack['train'][train_id] = info_data  # S000007_P0003_T00
            train_id += 1
    assert len(information_pack['train']) + len(information_pack['dev']) + len(information_pack['test']) == len(
        information)

    for md in mode:
        np.save(f"./{args.dataset}/{md}_info.npy", information_pack[md])
        # generate groundtruth stm for evaluation
        generate_gt_stm(information_pack[md], f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
    # update the total gloss dict
    sign_dict_update(sign_dict, information)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# default arguments for each dataset, used when not overridden on the command line
DATASET_DEFAULTS = {
    'phoenix2014': {
        'dataset_root': '../dataset/phoenix2014/phoenix-2014-multisigner',
        'annotation_prefix': 'annotations/manual/{}.corpus.csv',
    },
    'phoenix2014-T': {
        'dataset_root': '/disk1/dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T',
        'annotation_prefix': 'annotations/manual/PHOENIX-2014-T.{}.corpus.csv',
    },
    'CSL-Daily': {
        'dataset_root': '/sda/home/guozihang/guozihang/CSL-Daily/sentence/frames_512x512',
        'target_path': '/sda/home/guozihang/guozihang/CSL-Daily/CSL-Daily_256x256px',
        'annotation_file': 'video_map.txt',
        'split_file': 'split_1.txt',
    },
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='phoenix2014', choices=list(DATASET_DEFAULTS.keys()),
                        help='dataset to preprocess, also used as the output directory name')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default=None,
                        help='annotation prefix, Phoenix2014 / Phoenix2014-T only')
    parser.add_argument('--annotation-file', type=str, default=None,
                        help='annotation file, CSL-Daily only')
    parser.add_argument('--split-file', type=str, default=None,
                        help='split file, CSL-Daily only')
    parser.add_argument('--target-path', type=str, default=None,
                        help='target path for resized images, CSL-Daily only')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    # fill in per-dataset defaults
    for key, value in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.dataset == 'phoenix2014':
        process_phoenix(args, phoenix2014_csv2dict)
    elif args.dataset == 'phoenix2014-T':
        process_phoenix(args, phoenix2014_t_csv2dict)
    else:  # CSL-Daily
        process_csl_daily(args)
