"""Fuse the phoenix2014 and phoenix2014-T training bigarray memmap files
into a single fused memmap.

Converted from generate_ph_memmap.ipynb.
"""
import pickle

import numpy as np
from tqdm import tqdm

from gzhlaker import *

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ph_memmap_train_path = "/sda/data/guozihang/memmap/phoenix2014-bigarray-map-train"
ph_memmap_train_info_path = "/sda/data/guozihang/memmap/phoenix2014-train.pickle"
pht_memmap_train_path = "/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-bigarray-map-train"
pht_memmap_train_info_path = "/sda/data/guozihang/sign_dataset/pht_bigarray/phoenix2014-T-train.pickle"
fuse_memmap_train_path = "/sda/data/guozihang/fuse_ph_memmap"

H, W, C = 256, 256, 3

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
