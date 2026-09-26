# -*- encoding: utf-8 -*-
"""实验进程的启动、记录与状态查询。

训练一跑就是几小时,所以启动接口是「立刻返回 run_id」而不是阻塞等待:
每次启动都会在运行记录目录写下一条 JSON(命令、pid、日志路径、work_dir),
后续靠 ``get_run_status`` / ``tail_log`` 跟进,和自己在终端里跑
``script/run.sh`` 是同一件事,只是多了一份可查询的记录。

安全性:所有命令都以参数列表方式传给 subprocess(``shell=False``),不做
字符串拼接;``stop_run`` 只杀本服务记录过、且命令行对得上的进程。
"""

import json
import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .errors import McpToolError

STATUS_RUNNING = "running"
STATUS_EXITED = "exited"      # 正常结束(退出码 0)
STATUS_FAILED = "failed"      # 非 0 退出
STATUS_STOPPED = "stopped"    # 被 stop_run 主动停止
STATUS_UNKNOWN = "unknown"    # 进程已不在,但没有记录到退出码(服务重启过)

# 由工具参数决定,不接受用 extra_args 重复指定
RESERVED_FLAGS = ("--config", "--exp", "--work-dir", "--phase", "--device")

_STOP_GRACE_SECONDS = 15.0
_KILL_GRACE_SECONDS = 5.0
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-")
    return cleaned or "run"


class RunStore:
    """运行记录的持久化(一条记录一个 JSON 文件)。"""

    def __init__(self, runs_dir):
        self.runs_dir = Path(runs_dir)

    def _path(self, run_id):
        if not _RUN_ID_PATTERN.match(str(run_id)):
            raise McpToolError(f"非法的 run_id: {run_id!r}")
        return self.runs_dir / f"{run_id}.json"

    def new_id(self, label):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{_slug(label)}-{uuid.uuid4().hex[:4]}"

    def save(self, record):
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record["run_id"])
        handle, temp_name = tempfile.mkstemp(dir=str(self.runs_dir), prefix=".run_", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
            os.replace(temp_name, path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return path

    def load(self, run_id):
        path = self._path(run_id)
        if not path.is_file():
            raise McpToolError(
                f"找不到运行记录 {run_id!r};用 list_runs 查看本服务启动过的运行"
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise McpToolError(f"运行记录 {run_id!r} 已损坏: {exc}")

    def list(self, limit=None):
        if not self.runs_dir.is_dir():
            return []
        records = []
        for path in self.runs_dir.glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        records.sort(key=lambda item: item.get("started_at", ""), reverse=True)
        return records[:limit] if limit else records


class RunLauncher:
    """构造命令、启动进程并维护运行记录。"""

    def __init__(self, layout, store=None):
        self.layout = layout
        self.store = store or RunStore(layout.runs_dir)
        self._processes = {}  # run_id -> Popen(仅本进程启动的,用于准确回收退出码)

    # -------------------------------------------------------------- 命令构造

    def build_training_command(self, experiment, phase="train", work_dir=None,
                               device=None, extra_args=None):
        """拼出 ``python main.py --config configs/exp.yaml --exp <name> ...``。"""
        command = [
            self.layout.python,
            "main.py",
            "--config",
            str(self.layout.exp_config),
            "--exp",
            experiment,
            "--phase",
            phase,
        ]
        if work_dir:
            command += ["--work-dir", str(work_dir)]
        if device is not None:
            command += ["--device", str(device)]
        command += self._clean_extra_args(extra_args)
        return command

    def build_preprocess_command(self, dataset, dataset_root=None,
                                 process_image=False, extra_args=None):
        """拼出数据预处理命令(在 core/preprocess 下执行)。"""
        command = [self.layout.python, "dataset_preprocess.py", "--dataset", dataset]
        if dataset_root:
            command += ["--dataset-root", str(dataset_root)]
        if process_image:
            command.append("--process-image")
        command += self._clean_extra_args(extra_args)
        return command

    @staticmethod
    def _clean_extra_args(extra_args):
        """校验透传参数:必须是字符串列表,且不重复指定由工具参数决定的开关。

        这里要挡住的不只是 ``--exp`` 这类原样写法:argparse 还接受
        ``--exp=other`` 和 ``--ex other``(无歧义前缀缩写),两种都能绕过按
        整串比对的检查、把工具已经校验过的取值改掉。所以按 ``=`` 前面的开关名
        判断,并拒绝保留开关的任意前缀。
        """
        if not extra_args:
            return []
        if isinstance(extra_args, str) or not isinstance(extra_args, (list, tuple)):
            raise McpToolError("extra_args 必须是字符串列表,例如 ['--batch-size', '2']")
        cleaned = []
        for item in extra_args:
            if not isinstance(item, str):
                raise McpToolError(f"extra_args 里只能放字符串,收到 {item!r}")
            flag = item.split("=", 1)[0]
            if any(reserved.startswith(flag) for reserved in RESERVED_FLAGS):
                raise McpToolError(
                    f"{flag} 由工具参数决定(--exp/--phase/--work-dir/--device),"
                    "请不要放进 extra_args"
                )
            cleaned.append(item)
        return cleaned

    # -------------------------------------------------------------- 启动

    def launch_training(self, experiment, phase="train", work_dir=None, device=None,
                        extra_args=None, log_path=None):
        command = self.build_training_command(experiment, phase, work_dir, device, extra_args)
        return self._launch(
            kind="train" if phase == "train" else "eval",
            label=experiment,
            command=command,
            experiment=experiment,
            phase=phase,
            work_dir=work_dir,
            extra_args=list(extra_args or []),
            log_path=log_path,
            cwd=self.layout.core_dir,
        )

    def launch_preprocess(self, dataset, dataset_root=None, process_image=False,
                          extra_args=None, log_path=None):
        command = self.build_preprocess_command(dataset, dataset_root, process_image, extra_args)
        return self._launch(
            kind="preprocess",
            label=dataset,
            command=command,
            experiment=None,
            phase=None,
            work_dir=None,
            extra_args=list(extra_args or []),
            log_path=log_path,
            cwd=self.layout.core_dir / "preprocess",
        )

    def _launch(self, kind, label, command, experiment, phase, work_dir,
                extra_args, log_path, cwd):
        run_id = self.store.new_id(label)
        self.store.runs_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_path) if log_path else self.store.runs_dir / f"{run_id}.log"

        started_at = _utc_now()
        record = {
            "run_id": run_id,
            "kind": kind,
            "experiment": experiment,
            "phase": phase,
            "work_dir": work_dir,
            "command": command,
            "command_str": " ".join(str(part) for part in command),
            "cwd": str(cwd),
            "extra_args": extra_args,
            "log_path": str(log_path),
            "pid": None,
            "status": STATUS_RUNNING,
            "exit_code": None,
            "started_at": started_at,
            "finished_at": None,
        }

        with open(log_path, "ab") as log_stream:
            try:
                process = subprocess.Popen(  # noqa: S603 - 参数列表调用,不经 shell
                    command,
                    cwd=str(cwd),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env=_child_env(),
                )
            except FileNotFoundError:
                raise McpToolError(
                    f"启动失败:找不到解释器 {self.layout.python!r};"
                    "可用环境变量 OPENCSLR_PYTHON 指向训练环境里的 python"
                )
            except OSError as exc:
                raise McpToolError(f"启动失败: {exc}")

        record["pid"] = process.pid
        self._processes[run_id] = process
        self.store.save(record)
        record["log_tail"] = _tail_file(log_path, 20)
        return record

    # -------------------------------------------------------------- 状态

    def refresh(self, record):
        """刷新单个记录的状态(进程活着 -> running;已退出 -> 回填退出码)。

        状态真变了才落盘,避免 list_runs 轮询时反复写文件。
        """
        run_id = record.get("run_id")
        before = (record.get("status"), record.get("exit_code"))
        process = self._processes.get(run_id)
        if process is not None:
            code = process.poll()
            if code is not None:
                self._processes.pop(run_id, None)
                self._finalize(record, code)
            elif record.get("status") != STATUS_RUNNING:
                record["status"] = STATUS_RUNNING
        elif record.get("status") == STATUS_RUNNING and not _pid_alive(record.get("pid")):
            # 服务重启过或记录来自别的进程:只知道进程没了
            record["status"] = STATUS_UNKNOWN
            record["finished_at"] = record.get("finished_at") or _utc_now()
        if (record.get("status"), record.get("exit_code")) != before:
            self.store.save(record)
        return record

    def status(self, run_id):
        """返回刷新后的运行记录。"""
        record = self.refresh(self.store.load(run_id))
        record["pid_alive"] = _pid_alive(record.get("pid"))
        self.store.save(record)
        return record

    def list_runs(self, limit=20):
        records = []
        for record in self.store.list(limit=limit):
            self.refresh(record)
            record["pid_alive"] = _pid_alive(record.get("pid"))
            records.append(record)
        return records

    def _finalize(self, record, exit_code):
        record["exit_code"] = exit_code
        record["finished_at"] = record.get("finished_at") or _utc_now()
        if record.get("status") == STATUS_STOPPED:
            pass
        elif exit_code == 0:
            record["status"] = STATUS_EXITED
        else:
            record["status"] = STATUS_FAILED
        self.store.save(record)

    # -------------------------------------------------------------- 停止

    def stop(self, run_id, force=False):
        """停止一个本服务启动的进程(先 SIGTERM,必要时 SIGKILL)。

        Returns:
            dict: 刷新后的运行记录,附 ``note`` 说明停止结果。
        """
        record = self.refresh(self.store.load(run_id))
        pid = record.get("pid")
        if not _pid_alive(pid):
            record["note"] = "进程已不在运行,未发送信号"
            record["pid_alive"] = False
            return record

        _assert_our_process(record)
        _signal_process_group(pid, signal.SIGTERM)
        exited = self._wait_for_exit(record, _STOP_GRACE_SECONDS)

        if not exited and force:
            _signal_process_group(pid, signal.SIGKILL)
            exited = self._wait_for_exit(record, _KILL_GRACE_SECONDS)

        if exited:
            record["status"] = STATUS_STOPPED
            record["finished_at"] = record.get("finished_at") or _utc_now()
            record["note"] = "已停止(SIGKILL)" if force and record.get("exit_code") in (-9, 137) else "已停止"
            self.store.save(record)
        else:
            record["note"] = (
                f"已发送 SIGTERM 但进程 {_STOP_GRACE_SECONDS:.0f}s 内未退出;"
                "可传 force=True 强制终止"
            )
        record["pid_alive"] = _pid_alive(pid)
        return record

    def _wait_for_exit(self, record, timeout_seconds):
        """轮询等待进程退出;退出码来自 Popen,拿不到时按信号推断。"""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            process = self._processes.get(record["run_id"])
            if process is not None:
                code = process.poll()
                if code is not None:
                    self._processes.pop(record["run_id"], None)
                    record["exit_code"] = code
                    return True
            elif not _pid_alive(record.get("pid")):
                return True
            time.sleep(0.2)
        return False


# ------------------------------------------------------------------ 进程工具


def _pid_alive(pid):
    """pid 是否仍存在(存在但属于别人时也算 True)。"""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def _process_command(pid):
    """读取进程的完整命令行(用于确认要停的是不是自己启动的那个)。"""
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(int(pid))],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return result.stdout.strip() or None


def _assert_our_process(record):
    """确认 pid 对应的命令行确实是本记录启动的实验进程,否则拒绝发送信号。"""
    pid = record.get("pid")
    command = _process_command(pid)
    if command is None:
        raise McpToolError(f"读不到 pid {pid} 的命令行,拒绝停止")
    expected_script = "dataset_preprocess.py" if record.get("kind") == "preprocess" else "main.py"
    markers = [expected_script]
    if record.get("experiment"):
        markers.append(str(record["experiment"]))
    missing = [marker for marker in markers if marker not in command]
    if missing:
        raise McpToolError(
            f"拒绝停止 pid {pid}:命令行里没有 {missing},"
            f"实际命令行为 {command!r}。MCP 只停止自己启动的实验进程。"
        )


def _signal_process_group(pid, sig):
    """向进程组发信号(启动时用了 start_new_session,组 id 即 pid)。"""
    try:
        os.killpg(os.getpgid(int(pid)), sig)
    except ProcessLookupError:
        return
    except OSError:
        os.kill(int(pid), sig)


def _child_env():
    """子进程环境:让训练日志实时写盘,便于 tail_log 跟进。"""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _tail_file(path, lines):
    """读取文件末尾若干行(文件不存在时返回空字符串)。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            content = stream.readlines()
    except (OSError, FileNotFoundError):
        return ""
    return "".join(content[-lines:])
