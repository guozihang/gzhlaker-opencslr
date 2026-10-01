# -*- encoding: utf-8 -*-
"""训练中途热改超参数的控制通道(MCP 侧)。

训练进程通过 ``--control-file <path>`` 轮询一个 JSON 文件,本模块只负责:

- 按 revision 递增地写这个文件(写同一个 revision 不会被重复应用);
- 读回训练进程写的 ack(``<control-file>.ack.json``),看哪些键生效、哪些
  被拒绝以及理由。

文件格式与 ``core/utils/runtime_control.py`` 一一对应。哪些超参数能热改、
拒绝理由是什么,规则只存在于 core 那一侧——这里不复制第二份,否则两边迟早
说法不一致。要提前知道能改什么,用 ``get_hyperparameters``(它的数据同样来自
core 的真实规则表)。
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .errors import McpToolError

# 训练进程写回的确认文件后缀(与 core/utils/runtime_control.py 保持一致)
ACK_SUFFIX = ".ack.json"


def control_path(runs_dir, run_id):
    """某次运行的控制文件路径(放在运行记录目录里,不污染仓库配置)。"""
    return Path(runs_dir) / f"{run_id}.control.json"


def ack_path(control_file):
    return Path(str(control_file) + ACK_SUFFIX)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path):
    """读 JSON;不存在或损坏都返回 None(调用方据此判断「还没写/还没回」)。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def read_control(control_file):
    """控制文件当前内容(revision + overrides),没有则 None。"""
    return _read_json(control_file)


def read_ack(control_file):
    """训练进程写回的 ack,还没有则 None。"""
    return _read_json(ack_path(control_file))


def write_overrides(control_file, overrides):
    """把一组热改写成新的 revision。

    Args:
        control_file: 控制文件路径(训练进程正在轮询的那个)。
        overrides: 超参数名 -> 值;支持 ``optimizer_args.base_lr`` 这类点名写法。

    Returns:
        dict: 新写入的 ``revision`` 与文件路径。
    """
    if not isinstance(overrides, dict) or not overrides:
        raise McpToolError("overrides 必须是非空映射(超参数名 -> 值)")
    try:
        json.dumps(overrides, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise McpToolError(f"overrides 必须是可 JSON 序列化的值: {exc}")

    control_file = Path(control_file)
    previous = read_control(control_file) or {}
    try:
        revision = int(previous.get("revision") or 0) + 1
    except (TypeError, ValueError):
        revision = 1

    payload = {
        "revision": revision,
        "overrides": overrides,
        "issued_at": _utc_now(),
    }
    control_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(control_file, json.dumps(payload, ensure_ascii=False, indent=2))
    return {"revision": revision, "path": str(control_file), "payload": payload}


def state(control_file):
    """控制文件与 ack 的当前状态。

    ``pending``: 是否还有训练进程没确认的 revision(True 表示还没轮到/还在跑,
    False 表示最新 revision 已经处理过,None 表示无法判断)。
    """
    control = read_control(control_file)
    ack = read_ack(control_file)
    pending = None
    control_revision = (control or {}).get("revision")
    ack_revision = (ack or {}).get("revision")
    if control_revision is not None:
        if ack_revision is None:
            pending = True
        else:
            try:
                pending = int(control_revision) > int(ack_revision)
            except (TypeError, ValueError):
                pending = None
    return {
        "control_file": str(control_file),
        "ack_file": str(ack_path(control_file)),
        "control": control,
        "ack": ack,
        "pending": pending,
    }


def _atomic_write(path, text):
    """原子落盘:先写同目录临时文件再替换,训练进程不会读到半个 JSON。"""
    path = Path(path)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".control_", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
