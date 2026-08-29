"""制作 PHT (PHOENIX-2014-T) 的 memmap 文件

由 generate_pht_memmap.ipynb 转换而来。
"""
import os
import pickle

import numpy as np
from tqdm import tqdm

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider


# ---------------- helpers ----------------
def check_path(path):
    if not os.path.exists(path):
        os.system(f"mkdir -p {path}")


def get_file_name(path):
    """
    获取一个文件夹下的所有名称
    """
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files


def check_and_upload(file_list):
    if not file_list:
        return
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, 'https://oss-cn-huhehaote.aliyuncs.com', 'gzhlaker-experiment', region="cn-huhehaote")
    for local_path, oss_path, readme_path, readme_data in file_list:
        oss_exists = bucket.object_exists(oss_path)
        local_exists = os.path.exists(local_path)
        if local_exists and not oss_exists:
            bucket.put_object_from_file(oss_path, local_path)
            bucket.append_object(f'{readme_path}', 0, readme_data)
        elif oss_exists and not local_exists:
            bucket.get_object_to_file(oss_path, local_path)
        elif not oss_exists and not local_exists:
            print("无可用文件")


def generate_info(files, path):
    info = []
    start = 0
    length = 0
    for index, i in tqdm(enumerate(files)):
        file_path = os.path.join(path, i)
        data = np.load(file_path)
        length = data.shape[0]
        info.append({
            "path": file_path,
            "start": start,
            "end": start + length,
        })
        start = start + length
    return info


def generate_memmap(info, path):
    T = info[-1]["end"]
    H, W, C = 256, 256, 3
    fp = np.memmap(path, dtype='uint8', mode='w+', shape=(T, H, W, C))
    for index, item in tqdm(enumerate(info)):
        item_path, start, end = item["path"], item["start"], item["end"]
        data = np.load(item_path)
        fp[start:end] = data[:]
        if index % 500 == 0:
            fp.flush()
    fp.flush()


# ---------------- DIR Config ----------------
DATA_PATH = False
OUTPUT_PATH = True

if DATA_PATH:
    check_path("./data")

if OUTPUT_PATH:
    check_path("./output")

# ---------------- DATA OSS Config ----------------
DATA_FILE_LIST = []

# ---------------- 过程 ----------------
TRAIN_PATH = "/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/train/"
TEST_PATH = "/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/test/"
DEV_PATH = "/sda/data/guozihang/PHOENIX-2014-T-release-v3_cache/dev/"

TRAIN_FILES = get_file_name(TRAIN_PATH)
TEST_FILES = get_file_name(TEST_PATH)
DEV_FILES = get_file_name(DEV_PATH)

train_info = generate_info(TRAIN_FILES, TRAIN_PATH)
test_info = generate_info(TEST_FILES, TEST_PATH)
dev_info = generate_info(DEV_FILES, DEV_PATH)

with open("/sda/data/guozihang/phoenix2014-T-train.pickle", mode="wb") as f:
    pickle.dump(train_info, f)
with open("/sda/data/guozihang/phoenix2014-T-test.pickle", mode="wb") as f:
    pickle.dump(test_info, f)
with open("/sda/data/guozihang/phoenix2014-T-dev.pickle", mode="wb") as f:
    pickle.dump(dev_info, f)

generate_memmap(train_info, "/sda/data/guozihang/phoenix2014-T-bigarray-map-train")
generate_memmap(test_info, "/sda/data/guozihang/phoenix2014-T-bigarray-map-test")
generate_memmap(dev_info, "/sda/data/guozihang/phoenix2014-T-bigarray-map-dev")

# ---------------- OUTPUT OSS Config ----------------
OUTPUT_FILE_LIST = [
    ["./output/csldaily.pickle", "240902B/output/csldaily.pickle", "240902B/readme.txt", "这个文件是一个列表文件，每个列表包含一个三个字段，分别是'path','start','end'用于指定数据的路径，数据在 memmap 文件的开始位置，数据在 memmap 文件的结束为止"],
]
check_and_upload(OUTPUT_FILE_LIST)
