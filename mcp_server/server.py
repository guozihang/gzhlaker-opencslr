# -*- encoding: utf-8 -*-
"""MCP 工具定义(FastMCP)。

这是唯一依赖 `mcp` 包的模块;配置解析、运行管理、结果读取都在同包的其它
模块里实现,因此那些逻辑可以在不安装 MCP SDK 的情况下直接测试。

工具分三组:
    看清现状  list_experiments / get_experiment_config / list_options /
              resolve_experiment
    跑实验    create_experiment / launch_experiment / launch_preprocess /
              stop_run
    看结果    list_runs / get_run_status / get_results / list_artifacts /
              tail_log / list_work_dirs
"""

import functools
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):  # `python mcp_server/server.py`:补上包上下文
    # 直接执行文件时没有父包,下面的相对导入会以 "attempted relative import
    # with no known parent package" 失败(README 的快速开始给的就是这种用法)。
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "mcp_server"

try:  # MCP SDK >= 2.0(FastMCP 更名为 MCPServer)
    from mcp.server.mcpserver import MCPServer as _MCPServer
    from mcp.server.mcpserver.exceptions import ToolError as _ToolError
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as _MCPServer
    from mcp.server.fastmcp.exceptions import ToolError as _ToolError

from . import __version__
from .config import ExperimentConfig
from .errors import McpToolError
from .paths import RepoLayout
from .results import list_artifacts as _list_artifacts
from .results import list_work_dirs as _list_work_dirs
from .results import read_results as _read_results
from .results import tail_log as _tail_log
from .runs import STATUS_RUNNING, RunLauncher, RunStore

INSTRUCTIONS = """OpenCSLR 实验管理服务。

这个服务把仓库里的实验操作暴露成工具:查看有哪些实验、某个实验实际生效的
配置、启动/停止训练与评估、追踪运行状态、读取 WER 结果与 checkpoint。

约定(与上游 immc-lab/OpenCSLR 的配置一一对应):
- 一个实验 = core/configs/ 下一个扁平 YAML(如 vac.yaml),配置里的
  `model: models.build_function.build_vac` 是模型点号路径,命令行不再传 --model。
- core/configs/ 里同时放着数据集配置(phoenix2014.yaml 等):有 `model:` 的是
  实验,有 `dataset_root`/`dict_path` 的是数据集。
- 训练/评估都通过 `python main.py --config core/configs/<name>.yaml --phase ...`
  执行,工作目录必须是 core/(上游 map() 会去读 ./configs/<dataset>.yaml)。
- work_dir 既是结果/日志目录,也是 checkpoint 的文件名前缀。
- 上游 ConfigManager 没有嵌套键/取值校验;结果也不落盘 JSON,WER 只能从
  work_dir/log_*.log 里解析(get_results 会如实说明来源)。
- 长时间训练用 launch_experiment 启动(立刻返回 run_id),再用 get_run_status
  跟进;不要用同步方式等待训练结束。
"""

_STATE = {}
# 需要真正训练环境的解释器(装了 torch);解析配置只需要 PyYAML
_ENV_PYTHON = "OPENCSLR_PYTHON"


def _build_server():
    """构造 MCP 服务;1.x 的 FastMCP 还没有 version 参数,传了会炸。"""
    kwargs = {"instructions": INSTRUCTIONS}
    try:
        parameters = inspect.signature(_MCPServer.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - 拿不到签名时按不支持处理
        parameters = {}
    if "version" in parameters:
        kwargs["version"] = __version__
    return _MCPServer("opencslr", **kwargs)


mcp = _build_server()


def tool():
    """注册 MCP 工具,并把 McpToolError 翻译成 SDK 认得的 ToolError。

    这层翻译不是装饰:SDK 只把自家 ToolError 的 message 原样回给调用方,
    其它异常一律压成一句 `Error executing tool <name>`,调用方(智能体)就拿不到
    「实验不存在 / 配置校验失败」这类可据以行动的原因了。
    """

    def decorator(func):
        @functools.wraps(func)
        def synced(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except McpToolError as exc:
                raise _ToolError(str(exc)) from None

        return mcp.tool()(synced)

    return decorator


def configure(root=None):
    """初始化(或重置)服务状态,返回仓库布局。"""
    layout = RepoLayout.discover(root)
    layout.require_entries()
    store = RunStore(layout.runs_dir)
    config = ExperimentConfig(layout)
    launcher = RunLauncher(layout, store)
    _STATE["layout"] = layout
    _STATE["config"] = config
    _STATE["store"] = store
    _STATE["launcher"] = launcher
    # 清理上次服务留下的临时运行配置(只保留仍在跑的运行对应的那些)
    active = [
        record.get("run_id")
        for record in launcher.list_runs(limit=200)
        if record.get("status") == STATUS_RUNNING
    ]
    _STATE["swept_ephemeral"] = config.sweep_ephemeral_configs(active)
    return layout


def _component(name):
    if name not in _STATE:
        configure()
    return _STATE[name]


def _layout():
    return _component("layout")


# ====================================================================== 现状


@tool()
def get_server_info() -> Dict[str, Any]:
    """查看 MCP 服务连接的是哪个仓库、用哪个解释器跑实验。

    第一次调用建议先执行这个,确认 repo_root 与 configs_dir 是自己预期的仓库。
    """
    layout = _layout()
    info = layout.describe()
    info["missing_entries"] = layout.missing_entries()
    info["runs_dir_exists"] = os.path.isdir(info["runs_dir"])
    info["python_env_override"] = os.environ.get(_ENV_PYTHON)
    return info


@tool()
def list_experiments() -> Dict[str, Any]:
    """列出 configs/ 下所有可运行的实验(名称、模型点号路径、数据集、work_dir 等)。

    problems 字段列出配置有明显问题的实验(如 model 点号路径定位不到、数据集配置缺失)。
    """
    config = _component("config")
    return {
        "experiments": config.list_experiments(),
        "configs_dir": str(_layout().configs_dir),
    }


@tool()
def get_experiment_config(name: str) -> Dict[str, Any]:
    """查看某个实验「声明了什么」:exp 节及其引用的网络节、数据集节。

    这是配置文件里的原始内容(网络节打底、exp 节覆盖)。如果要看合并参数
    默认值之后「实际生效」的完整配置,请用 resolve_experiment。

    Args:
        name: 实验名,即 `--exp` 的取值(见 list_experiments)。
    """
    return _component("config").get_experiment_config(name)


@tool()
def list_options() -> Dict[str, Any]:
    """列出可选的实验名、网络名、数据集名,以及每个网络用到的模型名。

    新建实验(create_experiment)时的 model / dataset 取值必须来自这里。
    """
    return _component("config").list_options()


@tool()
def resolve_experiment(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """解析某个实验实际生效的配置,并做一次启动前的配置校验。

    跑的是 core/manager 里真实的初始化链(cli > yaml > 默认值),因此结论与
    真正启动时一致:ok=false 时 error 就是启动时会报出来的那句话。
    overrides 里可以放任何配置文件里能写的参数,用来预览「改这几个参数会
    怎样」——字典类配置(model_args/optimizer_args 等)按 YAML 语义深合并。

    Args:
        name: 实验名。
        overrides: 覆盖项,例如 {"num_epoch": 5, "model_args": {"use_bn": 0}}。
    """
    config = _component("config")
    if name not in config.experiment_names():
        raise McpToolError(
            f"实验 {name!r} 不存在;可用实验: {config.experiment_names()}"
        )
    response = config.resolve(name, overrides)
    if not response.get("ok"):
        return {
            "ok": False,
            "experiment": name,
            "error_type": response.get("error_type"),
            "error": response.get("error"),
            "hint": "修正 overrides 或配置文件后重试;不要带着这个错误启动训练",
        }
    effective = response["effective"]
    return {
        "ok": True,
        "experiment": name,
        "config_path": response.get("config_path"),
        "applied_overrides": response.get("overrides"),
        "key_settings": {
            key: effective.get(key)
            for key in ("model", "dataset", "decode_mode", "phase", "device",
                        "work_dir", "num_epoch", "batch_size", "random_seed")
        },
        "effective": effective,
    }


# ====================================================================== 调参

# 每类键在「能不能热改」上的说明模板(具体理由来自 core 的规则表)
_CHANGE_HINT_STARTUP = "启动前改:launch_experiment(overrides=...) 或 create_experiment"
_CHANGE_HINT_HOT = "可在训练中途修改(需要 core 提供运行时控制面)"


@tool()
def get_hyperparameters(name: str) -> Dict[str, Any]:
    """列出这个实验的全部超参数:当前生效值、类型/取值域、以及能否中途热改。

    调参前先看这个,别靠猜键名:清单与取值域来自真实的 main.py 参数表与
    ConfigManager 规则表;当前值来自真实的初始化链。

    返回里每条参数带:
    - current: 当前生效值
    - hot: 能否在训练中途改(True/False;None 表示当前 core 不支持热改)
    - hot_note / change_note: 热改时的说明,或不能热改的原因

    Args:
        name: 实验名(见 list_experiments)。
    """
    config = _component("config")
    if name not in config.experiment_names():
        raise McpToolError(
            f"实验 {name!r} 不存在;可用实验: {config.experiment_names()}"
        )

    schema = config.hyperparameters(name)
    if not schema.get("ok"):
        raise McpToolError(f"读取超参数清单失败: {schema.get('error')}")

    hot = schema.get("hot") or {}
    hot_keys = hot.get("keys")
    hot_set = set(hot_keys or [])
    descriptions = hot.get("descriptions") or {}
    reasons = hot.get("reasons") or {}

    resolution = schema.get("resolution") or {}
    effective = resolution.get("effective") if resolution.get("ok") else {}
    effective = effective or {}

    def annotate(key, value):
        """给一个键标上「能不能热改」。

        字典类键(optimizer_args / feeder_args)整体不在可热改列表里,但它的子键
        可能可以(optimizer_args.base_lr)。这时不能简单报「只能在启动前设置」——
        那会把调用方带偏,所以单列成 ``partial`` 并给出可热改的子键。
        """
        children = sorted(item for item in hot_set if item.startswith(f"{key}."))
        if hot_keys is None:
            return {
                "current": value,
                "hot": None,
                "hot_keys": [],
                "hot_note": None,
                "change_note": "当前 core 未提供运行时控制面,只能启动前改",
            }
        if key in hot_set:
            return {
                "current": value,
                "hot": True,
                "hot_keys": [key],
                "hot_note": descriptions.get(key),
                "change_note": _CHANGE_HINT_HOT,
            }
        if children:
            names = ", ".join(child.split(".", 1)[1] for child in children)
            return {
                "current": value,
                "hot": "partial",
                "hot_keys": children,
                "hot_note": ";".join(
                    filter(None, (descriptions.get(child) for child in children))
                ),
                "change_note": "部分子键可在训练中途改: {};其余键{}".format(
                    names, _CHANGE_HINT_STARTUP
                ),
            }
        return {
            "current": value,
            "hot": False,
            "hot_keys": [],
            "hot_note": descriptions.get(key),
            "change_note": "训练中途不可改: {};{}".format(
                reasons.get(key, "不在可热改列表里"), _CHANGE_HINT_STARTUP
            ),
        }

    arguments = []
    for item in schema.get("arguments", []):
        key = item["name"]
        arguments.append({**item, **annotate(key, effective.get(key, item.get("default")))})

    nested = {}
    for section_name, info in (schema.get("nested") or {}).items():
        nested[section_name] = {**info, **annotate(section_name, effective.get(section_name))}

    section = config.experiment_doc(name)
    return {
        "experiment": name,
        "model": section.get("model"),
        "dataset": section.get("dataset"),
        "config_path": str(config.config_path_for(name)),
        "resolution_ok": bool(resolution.get("ok")),
        "resolution_error": resolution.get("error"),
        "hot_keys": sorted(hot_set) if hot_keys is not None else None,
        "valid_optimizers": schema.get("valid_optimizers"),
        "arguments": arguments,
        "nested": nested,
        "how_to_change": {
            "启动前(推荐,会记进运行记录)": "launch_experiment(name, overrides={...})",
            "写进配置文件": "create_experiment(name, model, dataset, overrides={...})",
            "先预览": "resolve_experiment(name, overrides={...})",
            "训练中途": "上游 core 没有运行时控制面,改不了运行中的超参数",
        },
    }


# ====================================================================== 跑实验


@tool()
def create_experiment(
    name: str,
    model: str,
    dataset: str,
    overrides: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str] = None,
    device: Optional[str] = None,
    phase: Optional[str] = None,
    num_epoch: Optional[int] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """在 core/configs/ 下新建一个实验配置文件(上游=一实验一文件)。

    写入前会静态校验:model 点号路径能否在 core/ 里定位到、dataset 是否
    有对应的 configs/<dataset>.yaml、键是否都在 main.py 的参数表里。写完
    立即用上游真实的初始化链回读一次。

    Args:
        name: 实验名,同时是文件名 `<name>.yaml`,必须唯一。
        model: 模型点号路径,如 models.build_function.build_vac(见 list_options)。
        dataset: 数据集名,configs/ 下要有 <dataset>.yaml。
        overrides: 其它实验级参数,键名同 main.py 参数(下划线形式)。
        work_dir: 输出目录,建议以 / 结尾;checkpoint 会写成 `<work_dir>_best_model.pt`。
        device: GPU 序号,如 "0" 或 "0,1"。
        phase: train 或 test。
        num_epoch: 训练轮数。
        overwrite: 为 True 时覆盖同名配置文件。
    """
    return _component("config").create_experiment(
        name=name,
        model=model,
        dataset=dataset,
        overrides=overrides,
        work_dir=work_dir,
        device=device,
        phase=phase,
        num_epoch=num_epoch,
        overwrite=overwrite,
    )


@tool()
def launch_experiment(
    name: str,
    phase: str = "train",
    work_dir: Optional[str] = None,
    device: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """启动一个实验(默认训练),立刻返回 run_id。

    训练动辄数小时,所以这里不等结果:启动后可用 get_run_status 看进度、
    tail_log 看日志、get_results 看 WER。使用 GPU 前会先用真实的配置链校验
    一次(配置有问题会直接报错,不会浪费 GPU 时间)。

    传了 overrides 时,本次运行使用 core/configs 下一个临时配置(不改动实验
    自己的配置文件):覆盖项按 YAML 语义深合并,所以只改一个嵌套键不会丢掉
    同级的兄弟键。临时配置在进程结束后自动清理,快照留在运行记录目录里。
    上游 core 没有运行时控制面,所以启动之后不能再改超参数——要改就重新启动。

    Args:
        name: 实验名(见 list_experiments)。
        phase: train 训练 / test 只做评估(需配合 extra_args 里的 --load-weights)。
        work_dir: 覆盖配置里的输出目录;不传则用配置文件中的值。
        device: 覆盖使用的 GPU,如 "0" 或 "0,1"。
        extra_args: 透传给 main.py 的其它参数,逐个 token 写成列表,
            例如 ["--batch-size", "2", "--load-weights", "/path/best_model.pt"]。
            字典类参数不要写在这里(会被整块替换),请用 overrides。
        overrides: 超参数覆盖,键同 main.py 参数名,可嵌套,
            例如 {"optimizer_args": {"base_lr": 0.0001}, "num_epoch": 40}。
            先用 get_hyperparameters / resolve_experiment 看能改什么。
        dry_run: 为 True 时只返回将要执行的命令与生效值,不启动、不落盘。
    """
    config = _component("config")
    launcher = _component("launcher")
    store = _component("store")

    if name not in config.experiment_names():
        raise McpToolError(f"实验 {name!r} 不存在;可用实验: {config.experiment_names()}")
    if overrides is not None and not isinstance(overrides, dict):
        raise McpToolError("overrides 必须是映射(超参数名 -> 值)")

    run_id = store.new_id(name)
    # 没覆盖时直接用实验自己的 configs/<name>.yaml;有覆盖就落一份临时配置
    run_config_path = config.config_path_for(name)
    would_write = None
    materialized = None
    if overrides:
        if dry_run:
            # dry-run 不落盘:只算出「会写哪个临时配置」
            would_write = config.ephemeral_config_path(run_id)
            run_config_path = would_write
        else:
            materialized = config.materialize_run_config(name, overrides, run_id=run_id)
            run_config_path = Path(materialized["path"])

    # 启动前用真实配置链校验:有覆盖就校验临时配置本身,配置错误在这里就报,
    # 不等到占上 GPU。dry_run 没落盘,所以直接拿 overrides 去校验实验自己的
    # 配置文件(与 materialize 的深合并语义一致)。
    warning = None
    effective_work_dir = work_dir
    try:
        verdict = config.resolve(
            name, overrides=overrides, config_path=None if dry_run else run_config_path
        )
    except McpToolError as exc:
        verdict = None
        warning = f"跳过启动前校验({exc})"
    if verdict is not None and not verdict.get("ok"):
        _discard_materialized(config, materialized)
        raise McpToolError(
            f"实验 {name!r} 的配置校验失败: {verdict.get('error')};"
            "可先用 resolve_experiment 定位问题,不要带着这个错误启动训练"
        )
    if verdict is not None:
        effective_work_dir = work_dir or verdict["effective"].get("work_dir")

    command = launcher.build_training_command(
        name, phase, work_dir, device, extra_args, config_path=run_config_path,
    )
    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "experiment": name,
            "phase": phase,
            "command": command,
            "command_str": " ".join(command),
            "cwd": str(_layout().core_dir),
            "work_dir": effective_work_dir,
            "overrides": overrides or {},
            "config_path": str(run_config_path),
            "would_write_config": str(would_write) if would_write else None,
            "warning": warning,
        }

    if work_dir:
        _ensure_directory(_layout().resolve_work_dir(work_dir))

    record = launcher.launch_training(
        name, phase=phase, work_dir=work_dir, device=device, extra_args=extra_args,
        config_path=run_config_path, run_id=run_id,
        ephemeral_config=run_config_path if materialized else None,
        config_snapshot=(materialized or {}).get("snapshot"),
    )
    record["effective_work_dir"] = effective_work_dir
    record["warning"] = warning
    record["overrides"] = (materialized or {}).get("overrides") or overrides or {}
    _component("store").save(record)
    summary = _run_summary(record)
    summary["overrides"] = record["overrides"]
    summary["config_snapshot"] = record.get("config_snapshot")
    return summary


def _discard_materialized(config, materialized):
    """启动前校验失败时,把已经写出的临时配置删掉,不留垃圾。"""
    if materialized:
        config.cleanup_ephemeral_config(materialized.get("path"))
        snapshot = materialized.get("snapshot")
        if snapshot:
            try:
                Path(snapshot).unlink()
            except OSError:
                pass


@tool()
def launch_preprocess(
    dataset: str,
    dataset_root: Optional[str] = None,
    process_image: bool = False,
    extra_args: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """启动数据集预处理(dataset_preprocess.py),立刻返回 run_id。

    生成的 memmap/特征供 feeder_args.datatype 使用;跑之前确认磁盘空间。

    Args:
        dataset: 数据集名(见 list_options 或 dataset.yaml)。
        dataset_root: 原始数据目录;不传则用 dataset.yaml 里的默认值。
        process_image: 是否同时处理原始图像(耗时很长)。
        extra_args: 透传给 dataset_preprocess.py 的其它参数(逐个 token)。
        dry_run: 为 True 时只返回命令不启动。
    """
    datasets = _component("config").list_options()["datasets"]
    if dataset not in datasets:
        raise McpToolError(f"dataset {dataset!r} 不在 dataset.yaml 中;可用: {datasets}")

    launcher = _component("launcher")
    command = launcher.build_preprocess_command(dataset, dataset_root, process_image, extra_args)
    if dry_run:
        return {
            "dry_run": True,
            "dataset": dataset,
            "command": command,
            "command_str": " ".join(command),
            "cwd": str(_layout().core_dir / "preprocess"),
        }
    record = launcher.launch_preprocess(
        dataset, dataset_root=dataset_root, process_image=process_image, extra_args=extra_args
    )
    return _run_summary(record)


@tool()
def stop_run(run_id: str, force: bool = False) -> Dict[str, Any]:
    """停止由 launch_experiment / launch_preprocess 启动的进程。

    先发 SIGTERM,超过等待时间仍不退出时可用 force=True 发 SIGKILL。
    只会停止本服务记录过、且命令行对得上的进程。

    Args:
        run_id: 运行 id(见 list_runs)。
        force: 15 秒内没退出时是否强制终止。
    """
    return _component("launcher").stop(run_id, force=force)


# ====================================================================== 看结果


@tool()
def list_runs(limit: int = 20) -> Dict[str, Any]:
    """列出本服务启动过的运行(最新的在前)及其当前状态。

    Args:
        limit: 最多返回多少条。
    """
    launcher = _component("launcher")
    records = launcher.list_runs(limit=max(1, min(int(limit or 20), 200)))
    return {
        "count": len(records),
        "runs": [_run_summary(record) for record in records],
        "runs_dir": str(_layout().runs_dir),
    }


@tool()
def get_run_status(run_id: str, log_lines: int = 30) -> Dict[str, Any]:
    """查看一次运行的详细状态:进程是否还活着、日志末尾、已有结果与 checkpoint。

    这是跟进长训练的首选工具:一次调用就能知道「还在跑 / 挂了 / 出了多少 WER」。

    Args:
        run_id: 运行 id。
        log_lines: 返回日志末尾多少行。
    """
    launcher = _component("launcher")
    store = _component("store")
    record = launcher.status(run_id)
    summary = _run_summary(record)

    work_dir = record.get("effective_work_dir") or record.get("work_dir")
    summary["work_dir"] = work_dir
    summary["log_tail"] = _tail_log(
        _layout(), run_id=run_id, store=store, lines=max(1, min(int(log_lines or 30), 500))
    )["content"]

    if work_dir:
        try:
            results = _read_results(_layout(), work_dir)
            summary["results"] = results.get("summary")
            summary["results_note"] = results.get("note")
            artifacts = _list_artifacts(_layout(), work_dir)
            summary["checkpoints"] = [item["name"] for item in artifacts["checkpoints"][:5]]
        except McpToolError as exc:
            summary["results_note"] = f"读取结果失败: {exc}"

    return summary


@tool()
def get_results(work_dir: str) -> Dict[str, Any]:
    """读取某个 work_dir 下的实验结果(WER、样本统计等 JSON 内容)。

    Args:
        work_dir: 实验的输出目录,相对路径按 core/ 解析。
    """
    return _read_results(_layout(), work_dir)


@tool()
def list_artifacts(work_dir: str) -> Dict[str, Any]:
    """列出某个 work_dir 下的 checkpoint、结果文件与日志。

    checkpoint 既可能在目录里,也可能是 `<work_dir>_best_model.pt` 这种同级的
    前缀形式,这里会一起列出。

    Args:
        work_dir: 实验的输出目录。
    """
    return _list_artifacts(_layout(), work_dir)


@tool()
def tail_log(run_id: Optional[str] = None, work_dir: Optional[str] = None, lines: int = 100) -> Dict[str, Any]:
    """读取日志末尾若干行。

    run_id 与 work_dir 二选一:前者读 launch 工具重定向的运行日志(推荐,
    训练中的 stdout/stderr 都在这里),后者读 work_dir 下最新的 log_*.log。

    Args:
        run_id: 运行 id。
        work_dir: 实验输出目录。
        lines: 返回末尾多少行(1-2000)。
    """
    return _tail_log(
        _layout(),
        work_dir=work_dir,
        run_id=run_id,
        store=_component("store") if run_id else None,
        lines=lines,
    )


@tool()
def list_work_dirs(root: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """找出已有的实验输出目录(含结果文件的目录),用于回顾跑过哪些实验。

    Args:
        root: 搜索起点;默认查 core/work_dir 与仓库根下的 work_dir。
            共享存储上的目录(如 /sda/...)需显式传入。
        limit: 最多返回多少个目录。
    """
    return _list_work_dirs(_layout(), root=root, limit=max(1, min(int(limit or 50), 500)))


# ====================================================================== 内部


def _run_summary(record):
    """挑出运行记录里对调用方有用的字段。"""
    return {
        "run_id": record.get("run_id"),
        "kind": record.get("kind"),
        "experiment": record.get("experiment"),
        "phase": record.get("phase"),
        "status": record.get("status"),
        "pid": record.get("pid"),
        "pid_alive": record.get("pid_alive"),
        "exit_code": record.get("exit_code"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "work_dir": record.get("work_dir"),
        "command_str": record.get("command_str"),
        "config_path": record.get("config_path"),
        "config_snapshot": record.get("config_snapshot"),
        "log_path": record.get("log_path"),
        "note": record.get("note"),
        "warning": record.get("warning"),
        "log_tail": record.get("log_tail"),
    }


def _ensure_directory(path):
    """确保输出目录存在(训练自己也会建,这里提前建好便于早发现问题)。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise McpToolError(f"无法创建输出目录 {path}: {exc}")


if __name__ == "__main__":  # `python mcp_server/server.py --root .` 也能启动
    # 直接执行时本模块叫 __main__;先把它登记成 mcp_server.server,__main__.py
    # 里的 `from .server import configure, mcp` 才会复用同一个服务实例,而不是
    # 再导入一遍、把工具注册到第二个对象上。
    sys.modules.setdefault("mcp_server.server", sys.modules[__name__])

    from .__main__ import main

    raise SystemExit(main())
