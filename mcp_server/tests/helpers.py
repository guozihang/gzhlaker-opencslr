# -*- encoding: utf-8 -*-
"""测试用的临时仓库、假实验脚本与探针调用。

临时仓库刻意**照抄上游布局**:一个实验 = ``core/configs/<name>.yaml``,配置里
``model: models.build_function.build_vac`` 是点号路径;``core/configs/`` 里同时
放着数据集配置。``core/manager/`` 是真实的参数/配置管理器,所以临时仓库里的
探针跑的就是上游真正的解析逻辑,而不是一份测试专用的假规则。
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from mcp_server.paths import RepoLayout

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CORE = REPO_ROOT / "core"
REAL_CONFIGS = REAL_CORE / "configs"
CORE_PROBE = REPO_ROOT / "mcp_server" / "core_probe.py"

# 上游的实验配置(带 model:)与数据集配置(带 dataset_root/dict_path)
_EXPERIMENT_CONFIGS = ("baseline.yaml", "tlp.yaml", "vac.yaml")
_DATASET_CONFIGS = ("phoenix2014.yaml", "CSL-Daily.yaml")
# 探针只用到 load/get,但 config_manager 顶层 import 了 argument_manager
_MANAGER_MODULES = ("argument_manager.py", "config_manager.py")

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
    """建一个镜像上游布局的临时仓库,返回 RepoLayout。

    复制的内容:
      - ``core/main.py``        helpers.FAKE_MAIN(只打印 argv,支持 --fail/--sleep)
      - ``core/configs/``       真实的 vac/tlp/baseline + phoenix2014/CSL-Daily
      - ``core/manager/``       真实的 argument_manager.py + config_manager.py
      - ``core/models/build_function.py``
        真实文件,但只被 config.py 静态 ast 解析(校验 model 点号路径),从不导入,
        所以临时仓库里跑测试同样不需要 torch
      - ``core/preprocess/dataset_preprocess.py``  helpers.FAKE_PREPROCESS

    Args:
        with_configs: 为 False 时不拷贝任何配置(用于「configs/ 里什么都没有」的
            边界用例)。
    """
    root = Path(tempfile.mkdtemp(prefix="mcp_test_repo_"))
    core = root / "core"
    configs = core / "configs"
    configs.mkdir(parents=True)
    if with_configs:
        for name in _EXPERIMENT_CONFIGS + _DATASET_CONFIGS:
            shutil.copy(REAL_CONFIGS / name, configs / name)

    manager = core / "manager"
    manager.mkdir()
    for name in _MANAGER_MODULES:
        shutil.copy(REAL_CORE / "manager" / name, manager / name)

    models = core / "models"
    models.mkdir()
    shutil.copy(REAL_CORE / "models" / "build_function.py", models / "build_function.py")

    (core / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
    preprocess = core / "preprocess"
    preprocess.mkdir()
    (preprocess / "dataset_preprocess.py").write_text(FAKE_PREPROCESS, encoding="utf-8")
    return RepoLayout(root)


def run_probe(request, cwd=None, timeout=180):
    """按线上协议跑一次 ``core_probe.py`` 子进程,返回解析后的响应。

    与 ``config.py`` 调用探针的方式一致(请求走 stdin、响应是 stdout 最后一行
    JSON),所以这里验证的是真正被生产代码用到的协议,而不是直接 import 函数
    绕开进程边界。
    """
    proc = subprocess.run(
        [python_executable(), str(CORE_PROBE)],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        text=True,
        cwd=str(cwd or REAL_CORE),
        timeout=timeout,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    raise AssertionError(
        "探针没有返回 JSON(退出码 {}):stdout={!r} stderr={!r}".format(
            proc.returncode, proc.stdout, proc.stderr
        )
    )


def remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)


def force_kill_record(record):
    """兜底收尾:环境不支持 ``ps`` 时 ``stop_run`` 会拒绝发信号,这里直接按 pid 清理。

    用例本身不该因为沙箱差异留下还在 sleep 的假训练进程。
    """
    pid = record.get("pid")
    if not pid:
        return
    for attempt in (
        lambda: os.killpg(os.getpgid(int(pid)), signal.SIGKILL),
        lambda: os.kill(int(pid), signal.SIGKILL),
    ):
        try:
            attempt()
            return
        except Exception:
            continue


def python_executable():
    return sys.executable or "python3"


def write_json(path, payload):
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
