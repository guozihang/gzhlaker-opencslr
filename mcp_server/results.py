# -*- encoding: utf-8 -*-
"""实验结果、日志与 checkpoint 的读取。

**上游 core 不落盘任何结果 JSON**,所以这里的指标来自日志文本解析(这是
MCP 层对上游能力的如实适配,不是 core 的改动):

- ``<work_dir>/log_<时间戳>.log``           loguru 文件日志。上游把
  ``{"Dev": ..}`` / ``{"Test": ..}`` / ``Best_dev: .., Epoch : ..``
  写进日志(训练流程)或 ``Dev WER: ..`` / ``Test WER: ..``(test 阶段)。
- ``<work_dir>_best_model.pt``              最佳模型;注意 work_dir 同时被当作
  文件名前缀(见 ExperimentManager.run_train),所以轨迹记录形如
  ``<work_dir>dev_19.20_epoch5_model.pt``

如果目录里恰好有 ``experiment_result*.json`` / ``sample_statistics*.json``
(本仓库旧分支或用户自己的脚本写的),也会一并读出来 —— 属于额外福利,上游本身
不会生成。

因此 work_dir 既可能是目录(日志),也可能是文件前缀(checkpoint),这里两种都查。
"""

import glob
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .errors import McpToolError

_RESULT_GLOBS = ("experiment_result*.json", "sample_statistics*.json")
_CHECKPOINT_GLOBS = ("*_model.pt",)
_MAX_WALK_DEPTH = 3

# 上游日志里的 WER 行(loguru 会把 dict 原样 repr 出来)
_WER_LINE_PATTERNS = (
    ("dev", re.compile(r"[\"']Dev[\"']\s*:\s*([0-9]+(?:\.[0-9]+)?)")),
    ("test", re.compile(r"[\"']Test[\"']\s*:\s*([0-9]+(?:\.[0-9]+)?)")),
    ("dev", re.compile(r"Dev WER:\s*([0-9]+(?:\.[0-9]+)?)")),
    ("test", re.compile(r"Test WER:\s*([0-9]+(?:\.[0-9]+)?)")),
)
_BEST_LINE_PATTERN = re.compile(
    r"Best_dev:\s*([0-9]+(?:\.[0-9]+)?),\s*Epoch\s*:\s*(\d+)")
_EPOCH_LINE_PATTERN = re.compile(r"Epoch (\d+) costs")
_HISTORY_LIMIT = 20


def read_results(layout, work_dir):
    """读取 work_dir 下的结果。

    上游没有结果 JSON,所以主要指标从 ``log_*.log`` 里解析;若目录里存在
    ``experiment_result*.json`` / ``sample_statistics*.json`` 也会读出来。

    Args:
        layout: RepoLayout。
        work_dir: 配置里写的 work_dir(相对路径按 core/ 解析)。

    Returns:
        dict: ``work_dir``、``exists``、``results``(JSON 文件)、``summary``
        (关键指标)、``source``(指标来源)、``log_file``。
    """
    directory = layout.resolve_work_dir(work_dir)
    payload = {
        "work_dir": str(directory),
        "exists": directory.is_dir(),
        "results": {},
        "summary": {},
    }
    if not directory.is_dir():
        payload["note"] = (
            f"目录不存在: {directory};如果实验还没跑过,先 launch_experiment,"
            "或用 list_work_dirs 找已有的实验目录"
        )
        return payload

    for pattern in _RESULT_GLOBS:
        for path in sorted(directory.glob(pattern)):
            payload["results"][_split_key(path.name)] = _read_json(path)

    summary = {}
    for key, content in payload["results"].items():
        if not isinstance(content, dict) or "error" in content:
            continue
        if key.startswith("experiment_result"):
            summary[key] = {
                field: content.get(field)
                for field in ("split", "wer", "status", "total_samples",
                              "skip_rate", "timestamp", "experiment_name", "seed")
                if field in content
            }
        else:
            summary[key] = {
                field: content.get(field)
                for field in ("split", "total_samples", "num_successful",
                              "num_skipped", "num_failed", "skip_rate", "status")
                if field in content
            }

    log_file = _newest_log(directory)
    if log_file is not None:
        metrics = parse_log_metrics(log_file)
        payload["log_file"] = str(log_file)
        payload["summary"]["from_log"] = metrics
        payload["source"] = "log+json" if summary else "log"
    else:
        payload["source"] = "json" if summary else None

    payload["summary"].update(summary)
    if not payload["results"] and log_file is None:
        payload["note"] = (
            f"{directory} 下既没有 log_*.log 也没有结果 JSON;"
            "实验可能还没启动,或用 list_work_dirs 找找别的目录"
        )
    elif not payload["results"]:
        payload["note"] = (
            "上游 core 不落盘结果 JSON;上面的 from_log 是从日志文本解析出来的,"
            "权威指标仍在日志里(tail_log 可看原文)"
        )
    return payload


def _newest_log(directory):
    """目录下最新的 log_*.log;没有则 None。"""
    candidates = sorted(directory.glob("log_*.log"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_log_metrics(path):
    """从上游日志里解析 WER 与 epoch(纯文本,失败就返回 None 字段)。

    解析的是上游 ``LogManager.info`` 实际写出的那几行:
    ``{"Dev": 19.9}`` / ``{"Test": 20.1}`` / ``Best_dev: 19.90, Epoch : 3`` /
    ``Dev WER: 19.90`` / ``Test WER: 20.10`` / ``Epoch 3 costs ...``。
    """
    metrics = {
        "dev_wer": None,
        "test_wer": None,
        "best_dev_wer": None,
        "best_epoch": None,
        "last_epoch": None,
        "history": [],
        "lines_scanned": 0,
    }
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return metrics

    for line in text.splitlines():
        metrics["lines_scanned"] += 1
        best = _BEST_LINE_PATTERN.search(line)
        if best:
            metrics["best_dev_wer"] = float(best.group(1))
            metrics["best_epoch"] = int(best.group(2))
            continue
        epoch = _EPOCH_LINE_PATTERN.search(line)
        if epoch:
            metrics["last_epoch"] = int(epoch.group(1))
            continue
        for split, pattern in _WER_LINE_PATTERNS:
            found = pattern.search(line)
            if found:
                value = float(found.group(1))
                metrics["{}_wer".format(split)] = value
                metrics["history"].append({"split": split, "wer": value})
                break

    if len(metrics["history"]) > _HISTORY_LIMIT:
        metrics["history"] = metrics["history"][-_HISTORY_LIMIT:]
    return metrics



def list_artifacts(layout, work_dir):
    """列出 work_dir 相关的 checkpoint、结果文件与日志。"""
    directory = layout.resolve_work_dir(work_dir)

    checkpoints = {}
    for pattern in _checkpoint_globs(directory):
        for match in glob.glob(pattern):
            path = Path(match)
            checkpoints[str(path)] = _file_info(path)

    result_files = {}
    if directory.is_dir():
        for pattern in _RESULT_GLOBS:
            for path in sorted(directory.glob(pattern)):
                result_files[str(path)] = _file_info(path)

    logs = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.log")):
            info = _file_info(path)
            info["is_log"] = True
            logs.append(info)
    logs.sort(key=lambda item: item["mtime"], reverse=True)

    return {
        "work_dir": str(directory),
        "exists": directory.is_dir(),
        "checkpoints": sorted(checkpoints.values(), key=lambda item: item["mtime"], reverse=True),
        "result_files": sorted(result_files.values(), key=lambda item: item["path"]),
        "logs": logs,
        "latest_log": logs[0] if logs else None,
    }


def tail_log(layout, work_dir=None, run_id=None, store=None, lines=100):
    """读取日志末尾若干行。

    二选一:``run_id`` 读该次运行的重定向日志(launch 工具启动的),
    ``work_dir`` 读目录下最新的 ``log_*.log``(loguru 写的文件日志)。
    """
    lines = max(1, min(int(lines or 100), 2000))
    if run_id:
        if store is None:
            raise McpToolError("按 run_id 读日志需要运行记录目录")
        record = store.load(run_id)
        path = Path(record["log_path"])
        return {
            "source": "run",
            "run_id": run_id,
            "path": str(path),
            "lines": lines,
            "content": _tail(path, lines),
            "command": record.get("command_str"),
        }

    if not work_dir:
        raise McpToolError("需要提供 work_dir 或 run_id 之一")

    directory = layout.resolve_work_dir(work_dir)
    candidates = sorted(directory.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True) \
        if directory.is_dir() else []
    if not candidates:
        return {
            "source": "work_dir",
            "path": str(directory),
            "lines": lines,
            "content": "",
            "note": (
                f"{directory} 下没有 log_*.log;若实验由 launch_experiment 启动,"
                "请用 run_id 读日志(或 tail_log(run_id=...))"
            ),
        }
    newest = candidates[0]
    return {
        "source": "work_dir",
        "path": str(newest),
        "lines": lines,
        "content": _tail(newest, lines),
    }


def list_work_dirs(layout, root=None, limit=50):
    """找出已有的实验目录(含结果文件的目录)。

    用于「之前跑过哪些实验」这类问题;默认从 core/work_dir 与仓库根下的
    work_dir 开始找,也可以用 root 指定共享存储上的目录。
    """
    roots = []
    if root:
        roots.append(layout.resolve_work_dir(root))
    else:
        roots = [layout.core_dir / "work_dir", layout.root / "work_dir"]

    found = {}
    for base in roots:
        if not base.is_dir():
            continue
        for current, dirnames, filenames in os.walk(base):
            depth = len(Path(current).relative_to(base).parts)
            if depth >= _MAX_WALK_DEPTH:
                dirnames[:] = []
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            markers = [name for name in filenames if name.startswith("experiment_result")
                       or name.startswith("sample_statistics")]
            if markers:
                info = _file_info(Path(current))
                info.pop("size_mb", None)
                info["result_files"] = sorted(markers)
                found[current] = info

    directories = sorted(found.values(), key=lambda item: item["mtime"], reverse=True)
    return {
        "roots": [str(base) for base in roots],
        "count": len(directories),
        "work_dirs": directories[:limit],
    }


# ------------------------------------------------------------------ 内部工具


def _checkpoint_globs(directory):
    """返回 checkpoint 的 glob 模式(绝对路径)。

    ``work_dir`` 在 ExperimentManager 里同时被当作 checkpoint 的文件名前缀,
    所以两种写法都要查:
      - 目录形式 ``.../vac_smkd/``   -> ``.../vac_smkd/_best_model.pt``
      - 前缀形式 ``.../www2026_test_1`` -> ``.../www2026_test_1_best_model.pt``
    """
    return [
        os.path.join(str(directory), "*_model.pt"),
        str(directory) + "_*_model.pt",
    ]


def _split_key(filename):
    """``experiment_result_dev.json`` -> ``experiment_result_dev``。"""
    return filename[:-5] if filename.endswith(".json") else filename


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"JSON 解析失败: {exc}"}
    except OSError as exc:
        return {"error": f"读取失败: {exc}"}


def _file_info(path):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise McpToolError(f"读取文件信息失败: {path} ({exc})")
    return {
        "path": str(path),
        "name": path.name,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def _tail(path, lines):
    path = Path(path)
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            content = stream.readlines()
    except OSError:
        return ""
    return "".join(content[-lines:])
