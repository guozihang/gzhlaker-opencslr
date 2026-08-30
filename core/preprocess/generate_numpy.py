# # 生成缓存
"""临时脚本：将 phoenix2014/T/CSL-Daily 数据集的特征提取为 pickle 格式缓存。

注意：此文件仅用于一次性数据准备，包含硬编码路径，不应在训练中导入。
"""
import os
import torch
import torch.nn as nn
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from tqdm import tqdm

# ### 定义相关路径

def get_file_name(path):
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

import glob

a = get_file_name("/sda/data/guozihang/")

sample_name_dict = {}

from PIL import Image

import numpy
import cv2

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/CSL-Daily_256x256px", dir)
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/CSL-Daily_256x256px", dir, img)) for img in b]
    image_list_2 = [cv2.cvtColor(cv2.imread(os.path.join("/sda/data/guozihang/CSL-Daily_256x256px", dir, img_path)), cv2.COLOR_BGR2RGB) for img_path in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    break
    # np.save(f"/sda/data/guozihang/CSL-Daily_256x256px_cache/{dir}.npy", images)

images2 = np.split(images, images.shape[0], axis=0)
image_list_3 = [np.squeeze(im, axis=0) for im in images2]

image_list = [np.array(image.convert('RGB')) for image in image_list]

image_list[0][2,:,0], image_list_2[0][2,:,0], image_list_3[0][2,:,0]

a = get_file_name("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/")

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release-no-background/train/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/test/")
for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/test/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/test/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release_cache/test/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/PHOENIX-2014-T-release-v3-no-background/PHOENIX-2014-T/features/")
for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train/", dir)
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train/", dir, img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    if not os.path.exists("/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/"):
        os.system(f"mkdir -p /sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/")
    np.save(f"/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train/")
for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train/", dir)
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/train/", dir, img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    if not os.path.exists("/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/"):
        os.system(f"mkdir -p /sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/")
    np.save(f"/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/test/")
for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/test/", dir)
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/test/", dir,img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    if not os.path.exists("/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/test/"):
        os.system(f"mkdir -p /sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/test/")
    np.save(f"/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/test/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/dev/")
for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/dev/", dir)
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-256x256px/dev/", dir,img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    if not os.path.exists("/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/dev/"):
        os.system(f"mkdir -p /sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/dev/")
    np.save(f"/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/dev/{dir}.npy", images)

def read_img_to_numpy(path):
    img = cv2.imread(path)
    img_size = 256
    img = img.transpose([-1,0,1])
        
    return img

def append_img_to_file(img_path,file_name):
    img = read_img_to_numpy(img_path).reshape((-1))
    f = open(file_name, "ab+")
    for x in img:
        f.write(x.tobytes())
    f.close()

import cv2

a = read_img_to_numpy(os.path.join("/share/huaiwen_group/guozihang/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", "08April_2010_Thursday_tagesschau_default-11", "1", "08April_2010_Thursday_tagesschau.avi_pid0_fn000026-0.png"))

print(a.shape)

path = [
    [
        "../_Data/CSL-Daily/train_info.npy",
        "../_Data/CSL-Daily/dev_info.npy",
        "../_Data/CSL-Daily/test_info.npy"
    ],
    [
        "../_Data/CSL-Daily/train_min_info.npy",
        "../_Data/CSL-Daily/dev_min_info.npy",
        "../_Data/CSL-Daily/test_min_info.npy"
    ]
]

result_root = "/sda/home/guozihang/jupyter/_Output/J240601A"

# ### 定义相关方法

def load_annotation(path):
    """
    加载 tlp 处理的 phoenix2014 标注文件
    """
    annotation = np.load(path, allow_pickle=True).item()
    print(annotation[0])
    return annotation

train = load_annotation(path[0][0])
dev = load_annotation(path[0][1])
test = load_annotation(path[0][2])

train_min = load_annotation(path[1][0])
dev_min = load_annotation(path[1][1])
test_min = load_annotation(path[1][2])

len(train), len(train_min), len(dev), len(dev_min), len(test), len(test_min)

source = {}
for key in train:
    source[train[key]['fileid']] = [train[key]['fileid'], train[key]['num_frames'], train_min[key]['num_frames']]
for key in test:
    source[test[key]['fileid']] = [test[key]['fileid'], test[key]['num_frames'], test_min[key]['num_frames']]
for key in dev:
    source[dev[key]['fileid']] = [dev[key]['fileid'], dev[key]['num_frames'], dev_min[key]['num_frames']]

len(source)

def evenly_reduce_tensor(origin_length, target_length):
    """
    Reduce a tensor to a target length by removing elements evenly.

    Parameters:
    tensor (Tensor): The original tensor to be reduced.
    target_length (int): The target length of the tensor after reduction.

    Returns:
    Tensor: A new tensor reduced to the target length.
    """
    # 计算间隔
    interval = float(origin_length) / float(target_length)

    # 生成索引
    indices = (torch.arange(0, target_length) * interval).long()

    return indices

indices = {}
for key in source:
    indices[key] = evenly_reduce_tensor(source[key][1], source[key][2])

B = {}
for key in source:
    temp = torch.zeros(int(source[key][1]), dtype=torch.int8)
    temp[indices[key]] = 1
    B[key] = temp

kernel_sizes = ['K5', "P2", 'K5', "P2"]

import numpy as np

B["S000000_P0000_T00"]

C = {}
for key in B:
    left_pad = 0
    last_stride = 1
    total_stride = 1
    for layer_idx, ks in enumerate(kernel_sizes):
        if ks[0] == 'K':
            left_pad = left_pad * last_stride
            left_pad += int((int(ks[1]) - 1) / 2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride = total_stride * last_stride
    max_len = len(B[key])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad
    C[key] = torch.cat(
    (
        torch.zeros(left_pad),
        B[key],
        torch.zeros(max_len - len(B[key]) - left_pad)
    )
    , dim=0)

with open("B.pkl", mode="wb") as f:
    pickle.dump(C, f)

len(C["S007012_P0000_T00"]), len(B["S007012_P0000_T00"])

len(B['S005778_P0000_T00'])

len(C['S005778_P0000_T00'])

padded_video.shape

newB = torch.cat(
    (
        C['S005778_P0000_T00'][0].expand(10),
        C['S005778_P0000_T00'],
        C['S005778_P0000_T00'][-1].expand(max_len - C['S005778_P0000_T00'] - 10)
    )
    , dim=0)

C['S005778_P0000_T00'][0].expand(10)

C['S005778_P0000_T00'],

C['S005778_P0000_T00'][-1].expand(0)

a = get_file_name("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/train/")

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release-no-background_cache/train/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/test/")

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/test/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/test/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release-no-background_cache/test/{dir}.npy", images)

a = get_file_name("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/dev/")

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/dev/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/dev/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release-no-background_cache/dev/{dir}.npy", images)

a = np.load("/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/test/01April_2010_Thursday_heute-6704.npy")

a = get_file_name("/sda/data/guozihang/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/")

for dir in tqdm(a):
    b = os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1")
    b = sorted(get_file_name(b))
    b = b[int(torch.randint(0, 1, [1]))::1]
    image_list = [Image.open(os.path.join("/sda/data/guozihang/phoenix2014-release-no-background/phoenix-2014-multisigner/features/fullFrame-256x256px/train/", dir, "1",img)) for img in b]
    images = numpy.stack([np.array(image.convert('RGB')) for image in image_list], axis=0)
    # if not os.path.exists("/sda/data/guozihang/phoenix2014-release_cache/train/"):
    #     os.system(f"mkdir -p /sda/data/guozihang/phoenix2014-release_cache/train/")
    np.save(f"/sda/data/guozihang/phoenix2014-release-no-background_cache/train/{dir}.npy", images)
