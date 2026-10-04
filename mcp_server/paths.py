# -*- encoding: utf-8 -*-
"""OpenCSLR 仓库的目录布局。

只做纯路径计算,不导入 core/ 下的任何模块——MCP 服务必须能在没有 torch
的机器上启动(见 mcp_server/README.md 的部署说明)。
"""

import os
from pathlib import Path

# 覆盖仓库根目录(默认从本文件位置向上查找 core/configs/exp.yaml)
ENV_ROOT = "OPENCSLR_ROOT"
# 覆盖运行记录目录(默认 <repo>/.mcp_runs)
ENV_RUNS_DIR = "OPENCSLR_MCP_RUNS_DIR"
# 覆盖执行 core/main.py 用的解释器(默认本进程的解释器)
ENV_PYTHON = "OPENCSLR_PYTHON"

# 仓库根目录标志:向上查找 core/main.py(core/configs/ 在 require_entries 里另行检查)
_ROOT_MARKER = ("core", "main.py")


class RepoLayout:
    """仓库布局与关键入口路径。

    这是**上游 immc-lab/OpenCSLR 原样的布局**:一个实验 = ``core/configs/`` 下
    一个扁平 YAML(如 ``vac.yaml``),没有 exp.yaml/network.yaml 这种分节配置。

    所有实验命令都以 core/ 为工作目录执行 —— 上游
    ``ArgumentManager.map()`` 会去读 ``./configs/<dataset>.yaml``(相对进程 cwd),
    所以这个 cwd 是硬要求,不是习惯。
    """

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.core_dir = self.root / "core"
        self.configs_dir = self.core_dir / "configs"
        self.main_entry = self.core_dir / "main.py"
        self.preprocess_dir = self.core_dir / "preprocess"
        self.preprocess_entry = self.preprocess_dir / "dataset_preprocess.py"

    @classmethod
    def discover(cls, root=None):
        """按显式参数 -> 环境变量 -> 向上查找的顺序定位仓库。"""
        return cls(root if root is not None else find_repo_root())

    def missing_entries(self):
        """返回缺失的关键文件/目录列表(启动自检用)。"""
        entries = {
            "main_entry": self.main_entry,
        }
        missing = [f"{name} ({path})" for name, path in entries.items() if not path.is_file()]
        if not self.configs_dir.is_dir():
            missing.append(f"configs_dir ({self.configs_dir})")
        return missing

    def require_entries(self):
        """关键文件缺失时抛 FileNotFoundError。"""
        missing = self.missing_entries()
        if missing:
            raise FileNotFoundError(
                "仓库结构不完整,缺少: " + ", ".join(missing)
            )

    @property
    def runs_dir(self):
        """运行记录目录(MCP 自有的状态目录,不属于仓库配置)。"""
        override = os.environ.get(ENV_RUNS_DIR)
        if override:
            return Path(override).expanduser().resolve()
        return self.root / ".mcp_runs"

    @property
    def python(self):
        """执行 core/main.py 的解释器。

        默认与本进程相同;若 MCP 服务跑在没有 torch 的解释器下,用
        OPENCSLR_PYTHON 指向训练环境(如 conda 环境里的 python)。
        """
        return os.environ.get(ENV_PYTHON) or sys_default_python()

    def resolve_work_dir(self, work_dir):
        """把配置里/调用方给的 work_dir 解析为绝对路径。

        相对路径按 core/ 解析(训练进程的工作目录就是 core/);
        不做存在性检查,目录可能尚未创建。
        """
        path = Path(str(work_dir)).expanduser()
        if not path.is_absolute():
            path = self.core_dir / path
        return Path(os.path.normpath(str(path)))

    def describe(self):
        """布局摘要,给智能体确认「连的是哪个仓库」。"""
        return {
            "repo_root": str(self.root),
            "core_dir": str(self.core_dir),
            "configs_dir": str(self.configs_dir),
            "runs_dir": str(self.runs_dir),
            "python": self.python,
        }


def sys_default_python():
    """当前进程的解释器路径。"""
    import sys

    return sys.executable or "python3"


def find_repo_root(start=None):
    """从 start 逐级向上查找包含 core/main.py 的目录。

    Args:
        start: 起始目录;为 None 时优先读 OPENCSLR_ROOT,否则用本文件所在目录。

    Raises:
        FileNotFoundError: 向上找不到仓库根目录时抛出。
    """
    if start is None:
        override = os.environ.get(ENV_ROOT)
        if override:
            root = Path(override).expanduser().resolve()
            if not (root / Path(*_ROOT_MARKER)).is_file():
                raise FileNotFoundError(
                    f"{ENV_ROOT}={override} 下找不到 core/main.py,"
                    "请检查环境变量是否指向 OpenCSLR 仓库根目录"
                )
            return root
        start = Path(__file__).resolve().parent

    current = Path(start).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / Path(*_ROOT_MARKER)).is_file():
            return candidate
    raise FileNotFoundError(
        f"从 {current} 向上未找到 core/main.py;"
        f"可用环境变量 {ENV_ROOT} 指定 OpenCSLR 仓库根目录"
    )
