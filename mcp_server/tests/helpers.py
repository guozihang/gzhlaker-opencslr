# -*- encoding: utf-8 -*-
"""测试用的临时仓库与假训练脚本。"""

import os
import shutil
import signal
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

# 会真的轮询控制文件的假训练脚本。它用的就是仓库里真实的
# core/utils/runtime_control.py,因此能端到端验证整条介入链路:
# MCP 写控制文件 -> 训练进程应用 -> 写回 ack -> MCP 读回。
CONTROL_FAKE_MAIN = '''# -*- encoding: utf-8 -*-
import sys, time, types
from utils.runtime_control import RuntimeControl

argv = sys.argv
control_path = argv[argv.index("--control-file") + 1] if "--control-file" in argv else None
iterations = int(argv[argv.index("--iterations") + 1]) if "--iterations" in argv else 200


class FakeOptimizer:
    def __init__(self, lr):
        self.param_groups = [{"lr": lr, "weight_decay": 0.0}]


class FakeScheduler:
    def __init__(self, lr):
        self.base_lrs = [lr]
        self.milestones = {}


class FakeModel:
    def __init__(self, loss_weights):
        self.loss_weights = loss_weights

    def modules(self):
        return [self]


loss_weights = {"SeqCTC": 1.0}
arg = types.SimpleNamespace(
    optimizer_args={"base_lr": 0.001, "weight_decay": 0.0, "step": [5, 10]},
    num_epoch=10, save_interval=1, eval_interval=1, log_interval=200,
    print_log=True, loss_weights=loss_weights,
    feeder_args={"max_eval_frames": 100},
)
optimizer = FakeOptimizer(0.001)
scheduler = FakeScheduler(0.001)
model = FakeModel(loss_weights)

control = RuntimeControl(control_path, log=lambda message: print("[control]", message, flush=True))
control.bind(arg=arg, optimizer=optimizer, scheduler=scheduler, model=model)
print("[trainer] start", flush=True)
for step in range(iterations):
    control.poll(epoch=0, batch=step)
    print("[trainer] step=%d lr=%.6f" % (step, optimizer.param_groups[0]["lr"]), flush=True)
    time.sleep(0.1)
print("[trainer] done", flush=True)
'''


def make_temp_repo(with_configs=True):
    """建一个临时仓库,返回 RepoLayout。

    会拷一份真实的 core/manager 与 core/utils(都只含不导入 torch 的代码),
    这样临时仓库也能跑真实的配置解析与校验、并读到真实的热改规则表,而不是让
    用例依赖一份假规则。
    """
    root = Path(tempfile.mkdtemp(prefix="mcp_test_repo_"))
    core = root / "core"
    configs = core / "configs"
    configs.mkdir(parents=True)
    if with_configs:
        for name in ("exp.yaml", "network.yaml", "dataset.yaml"):
            shutil.copy(REAL_CONFIGS / name, configs / name)
    shutil.copytree(REAL_CONFIGS.parent / "manager", core / "manager",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # core/utils 没有 __init__.py(隐式命名空间包),但 runtime_control 就在里面:
    # core_probe 要从这里读「哪些超参数能热改」
    shutil.copytree(REAL_CONFIGS.parent / "utils", core / "utils",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (core / "main.py").write_text(FAKE_MAIN, encoding="utf-8")
    preprocess = core / "preprocess"
    preprocess.mkdir()
    (preprocess / "dataset_preprocess.py").write_text(FAKE_PREPROCESS, encoding="utf-8")
    return RepoLayout(root)


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
