# -*- encoding: utf-8 -*-
"""在 core/ 环境里解析实验配置(供 MCP 服务以子进程方式调用)。

为什么是子进程:配置的合并与校验规则只存在于 ``core/manager`` 里
(ArgumentManager 管 CLI/YAML 优先级与未知键检查,ConfigManager 管网络节
合并与嵌套键/取值校验)。MCP 层复刻一份规则迟早会漂移,所以这里直接跑
真实的初始化链,只把结果序列化成 JSON 回传。

这几步都不导入 torch(ArgumentManager/ConfigManager 只依赖标准库与
PyYAML),因此即使在没有 GPU 的机器上也能正常解析。

协议:
    stdin  <- {"exp": "...", "config": "<exp.yaml 绝对路径>", "overrides": {...}}
    stdout -> {"ok": true, ...} 或 {"ok": false, "stage": "...", "error": "..."}
"""

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


def _jsonable(value):
    """把配置值转成可 JSON 序列化的形式。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _argparse_actions(parser):
    """dest -> action 的映射(argparse 未提供公开的 action 枚举)。"""
    actions = {}
    for action in parser._actions:  # noqa: SLF001
        for option in action.option_strings:
            actions[option.lstrip("-").replace("-", "_")] = action
    return actions


def _split_overrides(parser, overrides):
    """把覆盖项分成「走 CLI 的」与「解析后深合并的」两类。

    标量/列表按 CLI 传入即可复用 ArgumentManager 的优先级(命令行 > YAML);
    而字典类配置(model_args / feeder_args / optimizer_args / loss_weights /
    wandb)必须深合并:一来 ``--optimizer-args`` 等参数在 argument_manager 里
    没有注册 ``type=json_dict``,CLI 传 JSON 字符串会被当成字符串整体替换;
    二来整块替换会丢掉 YAML 里继承的其它键(如 model_args.c2d_type)。

    Returns:
        (flag_overrides, deferred_overrides)
    """
    actions = _argparse_actions(parser)
    unknown = sorted(set(overrides) - set(actions))
    if unknown:
        raise ValueError(
            "未知配置项: {};exp.yaml 的键就是 argparse 参数名,"
            "--help 里没有的键写在这里会被 ConfigManager 拒绝".format(unknown)
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
    """执行一次解析,返回响应字典(不抛异常,失败也返回 ok=False)。"""
    try:
        return _resolve(request)
    except Exception as exc:
        return {
            "ok": False,
            "stage": "resolve",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _resolve(request):
    """真正干活的那一半;异常由 probe 统一转成 ok=False。"""
    core_dir = _core_dir()
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    # 不需要 chdir:配置路径都用绝对路径,dataset.yaml 由 ConfigManager 按
    # 配置所在目录查找。这样这个函数在测试里被直接调用也不会改变进程 cwd。

    from manager.argument_manager import ArgumentManager

    # 先解析空命令行,目的只是让 ArgumentManager 建好 PARSER(以便映射覆盖项)
    sys.argv = ["main.py"]
    ArgumentManager.init()
    parser = ArgumentManager.get_parser()

    config_path = os.path.abspath(request["config"])
    flag_overrides, deferred = _split_overrides(parser, request.get("overrides") or {})
    sys.argv = (
        ["main.py", "--config", config_path, "--exp", request["exp"]]
        + _override_flags(parser, flag_overrides)
    )

    from manager.config_manager import ConfigManager

    ArgumentManager.parse()
    ConfigManager.init()
    ArgumentManager.map(ConfigManager.get())

    # 字典类覆盖项:按 YAML 的语义深合并进最终参数,再用 ConfigManager 的
    # 校验规则复核一遍(CLI 路径不经过 ConfigManager,否则嵌套键/类型错误
    # 要等到训练时才暴露)。两个私有方法在这里被复用,是为了不把校验规则
    # 抄成第二份——规则只应该存在于 core/manager 里。
    args = ArgumentManager.get()
    for key, value in deferred.items():
        setattr(args, key, ArgumentManager._deep_merge(getattr(args, key), value))
    ConfigManager._validate(vars(args))

    return {
        "ok": True,
        "experiment": request["exp"],
        "config_path": config_path,
        "overrides": request.get("overrides") or {},
        "effective": _jsonable(vars(ArgumentManager.get())),
    }


def main():
    """从 stdin 读请求、向 stdout 写响应。"""
    try:
        raw = sys.stdin.read()
        request = json.loads(raw) if raw.strip() else {}
        if not request.get("exp"):
            raise ValueError("请求缺少 'exp' 字段")
        if not request.get("config"):
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
