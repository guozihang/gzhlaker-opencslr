# -*- encoding: utf-8 -*-
"""测试用的临时仓库与假训练脚本。"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from mcp_server.paths import RepoLayout

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIGS = REPO_ROOT / "core" / "configs"

# 假 main.py:按 --fail 参数决定退出码,其余情况睡眠,方便测试运行状态机
FAKE_MAIN = """# -*- encoding: utf-8 -*-
import sys, time

print("[fake-main] argv:", " ".join(sys.argv[1:]), flush=True)
if "--fail" in sys.argv:
    print("[fake-main] 故意失败", flush=True)
    sys.exit(3)
sleep_for = 0.0
if "--sleep" in sys.argv:
    sleep_for = float(sys.argv[sys.argv.index("--sleep") + 1])
if sleep_for:
    print("[fake-main] sleeping", sleep_for, flush=True)
    time.sleep(sleep_for)
print("[fake-main] done", flush=True)
"""

FAKE_PREPROCESS = """# -*- encoding: utf-8 -*-
import sys

print("[fake-preprocess] argv:", " ".join(sys.argv[1:]), flush=True)
"""


def make_temp_repo(with_configs=True):
    """建一个临时仓库,返回 RepoLayout。

    会拷一份真实的 core/manager(只含配置相关代码,不导入 torch),这样
    临时仓库也能跑真实的配置解析与校验,而不是让用例依赖一份假规则。
    """
    root = Path(tempfile.mkdtemp(prefix="mcp_test_repo_"))
    core = root / "core"
    configs = core / "configs"
    configs.mkdir(parents=True)
    if with_configs:
        for name in ("exp.yaml", "network.yaml", "dataset.yaml"):
            shutil.copy(REAL_CONFIGS / name, configs / name)
    shutil.copytree(REAL_CONFIGS.parent / "manager", core / "manager")
    (core / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
    preprocess = core / "preprocess"
    preprocess.mkdir()
    (preprocess / "dataset_preprocess.py").write_text(FAKE_PREPROCESS, encoding="utf-8")
    return RepoLayout(root)


def remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)


def python_executable():
    return sys.executable or "python3"


def write_json(path, payload):
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * 1024)
    return path


def env_without_root_override():
    """去掉可能影响定位的环境变量,保证用例独立。"""
    env = os.environ.copy()
    for key in ("OPENCSLR_ROOT", "OPENCSLR_MCP_RUNS_DIR", "OPENCSLR_PYTHON"):
        env.pop(key, None)
    return env
