"""制作 CSLDAILY 的 memmap 文件

由 generate_cd_memmap.ipynb 转换而来。
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


# ---------------- DIR Config ----------------
DATA_PATH = False
OUTPUT_PATH = True

if DATA_PATH:
    check_path("./data")

if OUTPUT_PATH:
    check_path("./output")

# ---------------- DATA OSS Config ----------------
DATA_FILE_LIST = []
check_and_upload(DATA_FILE_LIST)

# ---------------- 过程 ----------------
PATH = "/sda/data/guozihang/CSL-Daily_256x256px_cache/"
FILES = get_file_name(PATH)

info = []
start = 0
length = 0
for index, i in tqdm(enumerate(FILES)):
    path = os.path.join(PATH, i)
    data = np.load(path)
    length = data.shape[0]
    info.append({
        "path": path,
        "start": start,
        "end": start + length,
    })
    start = start + length

with open("./output/csldaily.pickle", mode="wb") as f:
    pickle.dump(info, f)

T, H, W, C = info[-1]["end"], 256, 256, 3
fp = np.memmap("/sda/data/guozihang/csldaily", dtype='uint8', mode='w+', shape=(T, H, W, C))
for index, item in tqdm(enumerate(info)):
    path, start, end = item["path"], item["start"], item["end"]
    data = np.load(path)
    fp[start:end] = data[:]
    if index % 500 == 0:
        fp.flush()
fp.flush()

# ---------------- OUTPUT OSS Config ----------------
OUTPUT_FILE_LIST = [
    ["./output/csldaily.pickle", "240902B/output/csldaily.pickle", "240902B/readme.txt", "这个文件是一个列表文件，每个列表包含一个三个字段，分别是'path','start','end'用于指定数据的路径，数据在 memmap 文件的开始位置，数据在 memmap 文件的结束为止"],
]
check_and_upload(OUTPUT_FILE_LIST)
