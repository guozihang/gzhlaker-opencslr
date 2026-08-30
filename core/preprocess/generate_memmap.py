"""制作 memmap 文件的合并脚本

由以下三个脚本合并而来：
- generate_pht_memmap.py  制作 PHT (PHOENIX-2014-T) 的 memmap 文件
- generate_ph_memmap.py   融合 phoenix2014 与 phoenix2014-T 的训练 memmap 文件
- generate_cd_memmap.py   制作 CSLDAILY (CSL-Daily) 的 memmap 文件

用法：
    python generate_memmap.py pht        # 制作 PHOENIX-2014-T 的 memmap 文件
    python generate_memmap.py ph         # 融合 phoenix2014 + phoenix2014-T 的训练 memmap
    python generate_memmap.py csldaily   # 制作 CSL-Daily 的 memmap 文件
"""
import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider


H, W, C = 256, 256, 3


# ---------------- helpers ----------------
def check_path(path):
    """检查路径是否存在，不存在则创建。"""
    if not os.path.exists(path):
        os.system(f"mkdir -p {path}")


def get_file_name(path):
    """获取一个文件夹下的所有文件名称。

    Args:
        path: 文件夹路径。

    Returns:
        文件名列表。
    """
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files


def check_and_upload(file_list):
    """检查本地与 OSS 上的文件是否存在，进行双向同步上传/下载。

    Args:
        file_list: 文件信息列表，每个元素为
            [local_path, oss_path, readme_path, readme_data]。
    """
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
    """生成包含每个文件起始位置和结束位置的信息列表。

    Args:
        files: 文件名列表。
        path: 文件所在文件夹路径。

    Returns:
        信息列表，每个元素包含 "path", "start", "end" 字段。
    """
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
    """根据信息列表将多个 .npy 文件合并写入一个 memmap 文件。

    Args:
        info: 信息列表，每个元素包含 "path", "start", "end" 字段。
        path: 输出的 memmap 文件路径。
    """
    T = info[-1]["end"]
    fp = np.memmap(path, dtype='uint8', mode='w+', shape=(T, H, W, C))
    for index, item in tqdm(enumerate(info)):
        item_path, start, end = item["path"], item["start"], item["end"]
        data = np.load(item_path)
        fp[start:end] = data[:]
        if index % 500 == 0:
            fp.flush()
    fp.flush()


# ---------------- PHT (PHOENIX-2014-T) ----------------
def generate_pht_memmap():
    """制作 PHOENIX-2014-T 的 memmap 文件。

    读取 train/test/dev 三个分片的 .npy 缓存文件，生成信息列表 pickle
    和对应的 memmap 文件，并上传至 OSS。
    """
    check_path("./output")

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

    OUTPUT_FILE_LIST = [
        ["./output/csldaily.pickle", "240902B/output/csldaily.pickle", "240902B/readme.txt", "这个文件是一个列表文件，每个列表包含一个三个字段，分别是'path','start','end'用于指定数据的路径，数据在 memmap 文件的开始位置，数据在 memmap 文件的结束为止"],
    ]
    check_and_upload(OUTPUT_FILE_LIST)


# ---------------- PH (fuse phoenix2014 + phoenix2014-T) ----------------
def generate_ph_memmap():
    """融合 phoenix2014 与 phoenix2014-T 的训练集 memmap 文件。

    将两个数据集的训练帧序列拼接到一个连续的帧索引空间中，生成融合后的
    memmap 文件及对应的信息列表。
    """
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    ph_memmap_train_path = "/sda/data/guozihang/memmap/phoenix2014-bigarray-map-train"
    ph_memmap_train_info_path = "/sda/data/guozihang/memmap/phoenix2014-train.pickle"
    pht_memmap_train_path = "/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-bigarray-map-train"
    pht_memmap_train_info_path = "/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-train.pickle"
    fuse_memmap_train_path = "/sda/data/guozihang/fuse_ph_memmap"

    # ---------------------------------------------------------------------------
    # Load sequence info: list of {"path", "start", "end"} dicts
    # ---------------------------------------------------------------------------
    with open(ph_memmap_train_info_path, mode="rb") as f:
        ph_train_info = pickle.load(f)
    with open(pht_memmap_train_info_path, mode="rb") as f:
        pht_train_info = pickle.load(f)

    print(f"ph sequences: {len(ph_train_info)}, last: {ph_train_info[-1]}")
    print(f"pht sequences: {len(pht_train_info)}")

    # ---------------------------------------------------------------------------
    # Map the source memmaps (read-only)
    # ---------------------------------------------------------------------------
    T = ph_train_info[-1]["end"]
    ph_fp = np.memmap(ph_memmap_train_path, dtype="uint8", mode="r", shape=(T, H, W, C))

    T = pht_train_info[-1]["end"]
    pht_fp = np.memmap(pht_memmap_train_path, dtype="uint8", mode="r", shape=(T, H, W, C))

    # ---------------------------------------------------------------------------
    # Offset phoenix2014-T frame ranges by the total number of phoenix2014
    # frames, so both datasets share one continuous frame-index space.
    # ---------------------------------------------------------------------------
    ph_offset = ph_train_info[-1]["end"]  # == 798990
    for item in tqdm(pht_train_info):
        item["start"] += ph_offset
        item["end"] += ph_offset

    # ---------------------------------------------------------------------------
    # Fused info + fused memmap
    # ---------------------------------------------------------------------------
    fuse_train_info = ph_train_info + pht_train_info

    T = fuse_train_info[-1]["end"]
    fu_fp = np.memmap(fuse_memmap_train_path, dtype="uint8", mode="w+", shape=(T, H, W, C))

    n_ph = len(ph_train_info)
    for index, item in tqdm(enumerate(fuse_train_info)):
        path, start, end = item["path"], item["start"], item["end"]
        if index < n_ph:
            fu_fp[start:end] = ph_fp[start:end]
        else:
            fu_fp[start:end] = pht_fp[start - ph_offset:end - ph_offset]
        if index % 500 == 0:
            fu_fp.flush()

    fu_fp.flush()


# ---------------- CSLDAILY (CSL-Daily) ----------------
def generate_csldaily_memmap():
    """制作 CSL-Daily 的 memmap 文件。

    读取 CSL-Daily 的 .npy 缓存文件，生成信息列表 pickle 和对应的
    memmap 文件，并上传至 OSS。
    """
    check_path("./output")

    PATH = "/sda/data/guozihang/CSL-Daily_256x256px_cache/"
    FILES = get_file_name(PATH)

    info = generate_info(FILES, PATH)

    with open("./output/csldaily.pickle", mode="wb") as f:
        pickle.dump(info, f)

    generate_memmap(info, "/sda/data/guozihang/csldaily")

    OUTPUT_FILE_LIST = [
        ["./output/csldaily.pickle", "240902B/output/csldaily.pickle", "240902B/readme.txt", "这个文件是一个列表文件，每个列表包含一个三个字段，分别是'path','start','end'用于指定数据的路径，数据在 memmap 文件的开始位置，数据在 memmap 文件的结束为止"],
    ]
    check_and_upload(OUTPUT_FILE_LIST)


# ---------------- CLI ----------------
def main():
    """命令行入口，根据参数选择制作对应数据集的 memmap 文件。"""
    parser = argparse.ArgumentParser(description="制作 memmap 文件的合并脚本（pht / ph / csldaily）")
    parser.add_argument(
        "command",
        choices=["pht", "ph", "csldaily"],
        help="pht: 制作 PHOENIX-2014-T 的 memmap 文件; ph: 融合 phoenix2014 + phoenix2014-T 的训练 memmap; csldaily: 制作 CSL-Daily 的 memmap 文件",
    )
    args = parser.parse_args()

    if args.command == "pht":
        generate_pht_memmap()
    elif args.command == "ph":
        generate_ph_memmap()
    else:
        generate_csldaily_memmap()


if __name__ == "__main__":
    main()
