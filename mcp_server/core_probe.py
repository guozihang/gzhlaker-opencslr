# -*- encoding: utf-8 -*-
"""在 core/ 环境里解析实验配置(供 MCP 服务以子进程方式调用)。

这里跑的是**上游 core 原样的初始化链**:``ArgumentManager`` 负责
CLI/YAML 优先级与「键必须是 argparse 参数」的检查,``ConfigManager`` 只做
``yaml.load`` + 字典合并(上游没有嵌套键/取值校验)。MCP 层不复制这些规则。

与上游约定有关的三件事:

1. 上游没有 ``--exp``,也**没有 network.yaml**:一个实验 = 一个扁平 YAML
   (``core/configs/vac.yaml``),里面的 ``model: models.build_function.build_vac``
   是点号路径,直接由配置提供,命令行不必再传 ``--model``。
2. ``ArgumentManager.map()`` 会去读 ``./configs/<dataset>.yaml``(相对**进程
   cwd**),所以探针必须在 ``core/`` 下运行 —— 由 ``config.py`` 用
   ``cwd=core_dir`` 保证。
3. 上游 ``manager/config_manager.py`` 顶层 ``import toml`` 但从不使用它;探针
   只用到 ``load``/``get``,所以在没装 toml 的环境里注入一个空桩,免得为了探针
   去装一个用不到的依赖(装了就走正常导入)。

协议:
    stdin  <- {"config": "<实验配置绝对路径>", "overrides": {...}}
    stdout -> {"ok": true, ...} 或 {"ok": false, "stage": "...", "error": "..."}

schema 模式(``{"mode": "schema", "config": ...}``)额外回传 main.py 的参数表。
"""

import contextlib
import copy
import io
import json
import os
import sys


def _core_dir():
    """定位 core/ 目录:优先当前工作目录,其次从本文件位置推断。"""
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, "main.py")) and os.path.isdir(
        os.path.join(cwd, "manager")
    ):
        return cwd
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"
    )


def _ensure_probe_dependencies():
    """上游 config_manager 顶层 import toml(用不到)。缺了就注入空桩。"""
    try:
        import toml  # noqa: F401
    except ImportError:
        import types

        sys.modules.setdefault("toml", types.ModuleType("toml"))


def _jsonable(value):
    """把配置值转成可 JSON 序列化的形式。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _last_line(text):
    """取多行文本的最后一行非空内容(argparse 的 usage 转储里,原因在最后一行)。"""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _deep_merge(base, override):
    """递归合并字典:嵌套字典逐层合并,其余由 override 覆盖。"""
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _argparse_actions(parser):
    """dest -> action 的映射(argparse 未提供公开的 action 枚举)。"""
    actions = {}
    for action in parser._actions:  # noqa: SLF001
        for option in action.option_strings:
            actions[option.lstrip("-").replace("-", "_")] = action
    return actions


def _split_overrides(parser, overrides):
    """把覆盖项分成「走 CLI 的」与「读盘后深合并的」两类。

    标量/列表走 CLI,直接复用上游 ``命令行 > YAML`` 的优先级;字典类配置
    (model_args / feeder_args / optimizer_args / loss_weights / wandb)必须深合并:
    上游 ``map()`` 是 ``set_defaults(**config)``,整块替换会丢掉 YAML 里已有的
    兄弟键(例如只想改 model_args.use_bn 却丢了 num_classes)。

    Returns:
        (flag_overrides, deferred_overrides)
    """
    actions = _argparse_actions(parser)
    unknown = sorted(set(overrides) - set(actions))
    if unknown:
        raise ValueError(
            "未知配置项: {};配置文件的键就是 main.py 的参数名,"
            "上游 ArgumentManager.map() 遇到不在 --help 里的键会直接报错".format(unknown)
        )
    flag_overrides = {}
    deferred = {}
    for key, value in overrides.items():
        if isinstance(value, dict):
            deferred[key] = value
        else:
            flag_overrides[key] = value
    return flag_overrides, deferred


def _override_flags(parser, overrides):
    """把标量/列表类覆盖项转成 CLI 参数。"""
    actions = _argparse_actions(parser)
    flags = []
    for key, value in overrides.items():
        action = actions[key]
        flag = action.option_strings[0]
        if value is None:
            continue
        if isinstance(value, bool):
            flags += [flag, "True" if value else "False"]
        elif isinstance(value, (list, tuple)):
            flags += [flag] + [str(item) for item in value]
        else:
            flags += [flag, str(value)]
    return flags


def probe(request):
    """执行一次请求,返回响应字典(不抛异常,失败也返回 ok=False)。"""
    try:
        if request.get("mode") == "schema":
            return _schema(request)
        return _resolve(request)
    except Exception as exc:
        return {
            "ok": False,
            "stage": "resolve",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _type_name(value):
    """把 argparse 的 type 参数转成可读名字。"""
    if value is None:
        return "str"
    if isinstance(value, (tuple, list)):
        return "|".join(_type_name(item) for item in value)
    return getattr(value, "__name__", str(value))


def _load_managers(config_path, overrides):
    """跑上游真实初始化链,返回 (ArgumentManager, 生效参数 dict)。

    顺序与 ``core/main.py`` 一致:ArgumentManager.init() → ConfigManager.init()
    → ArgumentManager.map()。区别只是 map 之前把字典类覆盖深合并进配置。
    """
    core_dir = _core_dir()
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    _ensure_probe_dependencies()

    from manager.argument_manager import ArgumentManager
    from manager.config_manager import ConfigManager

    config_path = os.path.abspath(config_path)

    # 先用空覆盖建 parser,才能把覆盖项映射成 CLI 参数
    sys.argv = ["main.py", "--config", config_path]
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            ArgumentManager.init()
    except SystemExit as exc:
        raise ValueError(
            _last_line(stderr.getvalue()) or f"命令行参数解析失败(argparse 退出码 {exc.code})"
        ) from None
    parser = ArgumentManager.get_parser()

    flag_overrides, deferred = _split_overrides(parser, overrides or {})
    sys.argv = ["main.py", "--config", config_path] + _override_flags(parser, flag_overrides)

    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            ArgumentManager.init()          # 带上覆盖项重新解析 CLI
            ConfigManager.init()            # 读实验配置
            config = dict(ConfigManager.get())
            for key, value in deferred.items():
                config[key] = _deep_merge(config.get(key), value)
            ArgumentManager.map(config)     # set_defaults + 重新 parse + 载入 dataset_info
    except SystemExit as exc:
        raise ValueError(
            _last_line(stderr.getvalue()) or f"命令行参数解析失败(argparse 退出码 {exc.code})"
        ) from None
    except AssertionError:
        raise ValueError(
            "配置里出现了 main.py 不认识的键(上游 map() 的断言);"
            "可用 --help 查看全部参数名"
        ) from None
    return ArgumentManager, config


def _resolve(request):
    """解析一次配置;异常由 probe 统一转成 ok=False。"""
    ArgumentManager, _ = _load_managers(request["config"], request.get("overrides") or {})
    return {
        "ok": True,
        "config_path": os.path.abspath(request["config"]),
        "overrides": request.get("overrides") or {},
        "effective": _jsonable(vars(ArgumentManager.get())),
    }


def _schema(request):
    """导出 main.py 的参数表 + 配置里各嵌套节的当前值。

    上游 ``ConfigManager`` **没有**嵌套键白名单/类型校验,所以这里只能给出
    「当前值」而给不出「允许哪些键、什么类型」—— ``allowed`` 一律为 null,
    由调用方如实转述,而不是假装有校验。
    """
    ArgumentManager = None
    if request.get("config"):
        ArgumentManager, _ = _load_managers(request["config"], request.get("overrides") or {})
    else:
        core_dir = _core_dir()
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        _ensure_probe_dependencies()
        from manager.argument_manager import ArgumentManager as _AM

        sys.argv = ["main.py"]
        _AM.init()
        ArgumentManager = _AM

    parser = ArgumentManager.get_parser()
    arguments = []
    for action in parser._actions:  # noqa: SLF001
        if not action.option_strings or action.dest == "help":
            continue
        names = [option.lstrip("-").replace("-", "_") for option in action.option_strings]
        arguments.append(
            {
                "name": names[0],
                "dest": action.dest,
                "flags": list(action.option_strings),
                "type": _type_name(action.type),
                "default": _jsonable(action.default),
                "choices": sorted(str(item) for item in action.choices) if action.choices else None,
                "nargs": action.nargs,
                "help": (action.help or "").strip() or None,
            }
        )

    # 字典类参数就是「嵌套节」;上游没有白名单,只能报当前值
    nested = {}
    resolution = None
    if request.get("config"):
        effective = vars(ArgumentManager.get())
        resolution = {"ok": True, "effective": _jsonable(effective)}
        for key, value in effective.items():
            if isinstance(value, dict) and value:
                nested[key] = {
                    "allowed": None,
                    "types": {},
                    "current": _jsonable(value),
                }

    return {
        "ok": True,
        "arguments": arguments,
        "nested": nested,
        "resolution": resolution,
        "valid_optimizers": None,
        "hot": {
            "keys": None,
            "descriptions": {},
            "reasons": {},
            "unknown_reason": None,
            "note": "上游 core 没有运行时控制面,训练中途不能热改超参数",
        },
    }


def main():
    """从 stdin 读请求、向 stdout 写响应。"""
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
        is_schema = request.get("mode") == "schema"
        if not request.get("config") and not is_schema:
            raise ValueError("请求缺少 'config' 字段")
        response = probe(request)
    except Exception as exc:
        response = {
            "ok": False,
            "stage": "resolve",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
