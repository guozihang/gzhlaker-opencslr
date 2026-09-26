# -*- encoding: utf-8 -*-
"""实验配置的读取、解析与创建。

仓库只有三个配置入口(core/configs/exp.yaml、network.yaml、dataset.yaml),
这里围绕它们提供智能体需要的操作:

- 看清有哪些实验、各自引用哪个网络/数据集(``list_experiments``)
- 拿到某个实验「声明了什么」以及「实际生效的配置是什么」
- 新建/更新实验节(``create_experiment``)

「实际生效的配置」不在这一层重算:优先级与校验规则只存在于 core/manager,
复刻一份必然漂移,所以委托给 ``core_probe.py`` 跑真实的初始化链。
"""

import ast
import copy
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from .errors import McpToolError

# 实验名:YAML 键,且不能以下划线开头(下划线节在本仓库里是 anchor 复用用的)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_ANCHOR_PREFIX = "_"

# exp.yaml 里供各实验复用的公共节;新建实验时优先继承它,保持与现有节一致的写法
_COMMON_SECTION = "_common_experiment"

_PROBE_PATH = Path(__file__).resolve().with_name("core_probe.py")
_PROBE_TIMEOUT_SECONDS = 180


class ExperimentConfig:
    """配置文件视图(每次调用都重新读盘,避免拿到过期配置)。"""

    def __init__(self, layout):
        self.layout = layout
        self._argument_names_cache = None
        self._argument_names_cache_key = None

    # ------------------------------------------------------------------ 读取

    def _load_yaml(self, path, label):
        try:
            text = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise McpToolError(f"{label} 不存在: {path}")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise McpToolError(f"{label} 不是合法 YAML: {exc}")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise McpToolError(f"{label} 顶层必须是映射(实验名 -> 配置节)")
        return data

    def exp_doc(self):
        return self._load_yaml(self.layout.exp_config, "exp 配置")

    def network_doc(self):
        return self._load_yaml(self.layout.network_config, "network 配置")

    def dataset_doc(self):
        return self._load_yaml(self.layout.dataset_config, "dataset 配置")

    def experiment_names(self):
        """全部可运行的实验名(排除下划线开头的 anchor 复用节)。"""
        return sorted(
            name for name in self.exp_doc() if not str(name).startswith(_ANCHOR_PREFIX)
        )

    def list_experiments(self):
        """实验总览:名称、引用的网络与数据集、关键设置。"""
        exp_doc = self.exp_doc()
        networks = self.network_doc()
        datasets = self.dataset_doc()
        items = []
        for name in sorted(k for k in exp_doc if not str(k).startswith(_ANCHOR_PREFIX)):
            section = exp_doc[name] if isinstance(exp_doc[name], dict) else {}
            network_name = section.get("network")
            dataset_name = section.get("dataset")
            problems = []
            if not network_name:
                problems.append("缺少 network 字段")
            elif network_name not in networks:
                problems.append(f"network {network_name!r} 未在 network.yaml 中定义")
            if not dataset_name:
                problems.append("缺少 dataset 字段")
            elif dataset_name not in datasets:
                problems.append(f"dataset {dataset_name!r} 未在 dataset.yaml 中定义")
            items.append(
                {
                    "name": name,
                    "network": network_name,
                    "dataset": dataset_name,
                    "phase": section.get("phase"),
                    "work_dir": section.get("work_dir"),
                    "device": section.get("device"),
                    "num_epoch": section.get("num_epoch"),
                    "problems": problems,
                }
            )
        return items

    def get_experiment_config(self, name):
        """某个实验「声明了什么」:exp 节、引用的网络节与数据集节。

        合并方式与 ConfigManager.load_experiment 一致:网络节打底,exp 节覆盖
        同名键,并去掉仅作引用用的 ``network`` 键(浅合并,嵌套字典不递归)。
        """
        exp_doc = self.exp_doc()
        if name not in exp_doc:
            raise McpToolError(
                f"实验 {name!r} 不存在于 {self.layout.exp_config};"
                f"可用实验: {self.experiment_names()}"
            )
        section = exp_doc[name]
        if not isinstance(section, dict):
            raise McpToolError(f"实验 {name!r} 的配置节必须是映射")

        network_name = section.get("network")
        dataset_name = section.get("dataset")
        networks = self.network_doc()
        datasets = self.dataset_doc()

        network_section = networks.get(network_name) or {}
        dataset_section = datasets.get(dataset_name) or {}

        merged = dict(network_section)
        merged.update(section)
        merged.pop("network", None)

        notes = []
        if network_name and network_name not in networks:
            notes.append(
                f"network {network_name!r} 未在 network.yaml 中定义;"
                f"可用: {sorted(k for k in networks if not str(k).startswith('_'))}"
            )
        if dataset_name and dataset_name not in datasets:
            notes.append(
                f"dataset {dataset_name!r} 未在 dataset.yaml 中定义;"
                f"可用: {sorted(k for k in datasets if not str(k).startswith('_'))}"
            )
        unknown = sorted(set(merged) - self.allowed_argument_names() - {"dataset"})
        if unknown:
            notes.append(f"以下键不是 main.py 的参数,启动时会被拒绝: {unknown}")

        return {
            "name": name,
            "config_path": str(self.layout.exp_config),
            "network": {"name": network_name, "config": network_section},
            "dataset": {"name": dataset_name, "config": dataset_section},
            "arguments": merged,
            "notes": notes,
        }

    def list_options(self):
        """可选项:网络、数据集、实验名,以及每个网络用到的模型名。"""
        networks = {
            name: section.get("model") if isinstance(section, dict) else None
            for name, section in self.network_doc().items()
            if not str(name).startswith(_ANCHOR_PREFIX)
        }
        datasets = sorted(
            k for k in self.dataset_doc() if not str(k).startswith(_ANCHOR_PREFIX)
        )
        return {
            "experiments": self.experiment_names(),
            "networks": networks,
            "datasets": datasets,
            "exp_config": str(self.layout.exp_config),
        }

    def allowed_argument_names(self):
        """main.py 接受的参数名(蛇形),从 argument_manager.py 静态提取。

        直接解析源码而不是导入,是为了在没装 torch 的机器上也能用;每次按
        文件 mtime 失效缓存,新增参数后无需重启 MCP 服务。
        """
        source_path = self.layout.core_dir / "manager" / "argument_manager.py"
        try:
            stat = source_path.stat()
            cache_key = (str(source_path), stat.st_mtime_ns)
        except FileNotFoundError:
            raise McpToolError(f"找不到参数定义文件: {source_path}")

        if self._argument_names_cache_key != cache_key:
            self._argument_names_cache = _extract_argument_names(source_path)
            self._argument_names_cache_key = cache_key
        return self._argument_names_cache

    # ------------------------------------------------------------------ 解析

    def resolve(self, name, overrides=None, timeout=_PROBE_TIMEOUT_SECONDS):
        """跑真实的初始化链,得到实际生效的配置与校验结论。

        Args:
            name: 实验名。
            overrides: 覆盖项(与 exp.yaml 同名的参数);字典类配置按 YAML
                的语义深合并。
            timeout: 子进程超时(秒)。

        Returns:
            dict: ``core_probe`` 的响应,``ok`` 为 False 时带 ``error`` 说明。
        """
        request = {
            "exp": name,
            "config": str(self.layout.exp_config),
            "overrides": overrides or {},
        }
        try:
            proc = subprocess.run(
                [self.layout.python, str(_PROBE_PATH)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                cwd=str(self.layout.core_dir),
                timeout=timeout,
            )
        except FileNotFoundError:
            raise McpToolError(
                f"找不到解释器 {self.layout.python!r};"
                "可用环境变量 OPENCSLR_PYTHON 指向训练环境里的 python"
            )
        except subprocess.TimeoutExpired:
            raise McpToolError(f"解析实验 {name!r} 超时(>{timeout}s)")

        response = _last_json_line(proc.stdout)
        if response is None:
            detail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise McpToolError(
                f"解析实验 {name!r} 的子进程没有返回结果(退出码 {proc.returncode}): {detail}"
            )
        return response

    # ------------------------------------------------------------------ 写入

    def create_experiment(
        self,
        name,
        network,
        dataset,
        overrides=None,
        work_dir=None,
        device=None,
        phase=None,
        num_epoch=None,
        overwrite=False,
    ):
        """新建(或整节替换)一个实验节。

        新节默认继承 ``_common_experiment``(与现有实验写法一致,后续改公共
        设置会一起生效);exp.yaml 是逐行追加的,其余实验节与注释保持原样。

        Args:
            name: 实验名,同时作为 ``--exp`` 的取值。
            network: network.yaml 中的网络名。
            dataset: dataset.yaml 中的数据集名。
            overrides: 其它实验级参数(键同 main.py 参数名),可为嵌套字典。
            work_dir: 输出目录;不传则沿用公共节里的值。
            device: GPU 序号(如 "0" 或 "0,1")。
            phase: train / test。
            num_epoch: 训练轮数。
            overwrite: 为 True 时整节替换已有实验(该节内的注释会丢失)。

        Returns:
            dict: 写入结果,含写入的配置节与解析校验结论。
        """
        self._validate_experiment_name(name)

        exp_doc = self.exp_doc()
        exists = name in exp_doc
        if exists and not overwrite:
            raise McpToolError(
                f"实验 {name!r} 已存在;如需整节替换请传 overwrite=True,"
                "或换一个实验名"
            )

        networks = self.network_doc()
        if network not in networks:
            raise McpToolError(
                f"network {network!r} 未在 network.yaml 中定义;"
                f"可用: {sorted(k for k in networks if not str(k).startswith('_'))}"
            )
        datasets = self.dataset_doc()
        if dataset not in datasets:
            raise McpToolError(
                f"dataset {dataset!r} 未在 dataset.yaml 中定义;"
                f"可用: {sorted(k for k in datasets if not str(k).startswith('_'))}"
            )

        fields = copy.deepcopy(overrides or {})
        if not isinstance(fields, dict):
            raise McpToolError("overrides 必须是映射(参数名 -> 值)")
        fields["network"] = network
        fields["dataset"] = dataset
        for key, value in (
            ("work_dir", work_dir),
            ("device", device),
            ("phase", phase),
            ("num_epoch", num_epoch),
        ):
            if value is not None:
                fields[key] = value

        allowed = self.allowed_argument_names() | {"network"}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise McpToolError(
                f"以下键不是 main.py 的参数: {unknown};"
                f"可用键: {sorted(allowed)}"
            )

        text = self.layout.exp_config.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        # 节的键名与 anchor 名不一定相同(如 `_common_experiment: &common_experiment`),
        # 所以用文本里真实的 anchor 名,而不是拿键名去拼
        common_anchor = _find_anchors(text).get(_COMMON_SECTION)
        block = self._render_section(name, fields, common_anchor)

        if exists:
            start, end = _section_line_range(text, name)
            new_text = text[:start] + block + text[end:]
        else:
            new_text = text + ("\n" if not text.endswith("\n\n") else "") + block

        # 先整份重新解析一遍:既校验 YAML 语法,也确认 anchor 引用可用
        try:
            parsed = yaml.safe_load(new_text)
        except yaml.YAMLError as exc:
            raise McpToolError(f"生成的新配置无法解析,已放弃写入: {exc}")
        if not isinstance(parsed, dict) or name not in parsed:
            raise McpToolError("生成的新配置缺少目标实验节,已放弃写入")

        _atomic_write(self.layout.exp_config, new_text)

        result = {
            "name": name,
            "config_path": str(self.layout.exp_config),
            "replaced_existing": bool(exists),
            "section": parsed[name],
            "inherits_common": _COMMON_SECTION in exp_doc,
        }
        # 用真实初始化链复核一次;失败也保留写入结果,由调用方决定是否回退
        verdict = self.resolve(name)
        result["validation"] = {
            "ok": verdict.get("ok"),
            "error": verdict.get("error"),
            "error_type": verdict.get("error_type"),
        }
        return result

    def _validate_experiment_name(self, name):
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            raise McpToolError(
                f"实验名 {name!r} 不合法:只能由字母开头,包含字母/数字/下划线/点/短横线"
            )

    def _render_section(self, name, fields, common_anchor):
        """生成实验节的 YAML 文本(带缩进与公共节继承)。

        Args:
            common_anchor: exp.yaml 里公共节的 anchor 名;为 None 时写出完整
                的独立节(不依赖 anchor)。
        """
        ordered = dict(
            _ordered_items(fields, ["network", "dataset", "work_dir", "device", "phase"])
        )
        # 整节一起 dump 再统一缩进:逐个键 dump 时 PyYAML 会把单键映射写成
        # 流式({network: x}),嵌进块映射里就成了非法 YAML
        body = yaml.safe_dump(
            ordered,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
        ).rstrip("\n")

        lines = [f"{name}:"]
        if common_anchor:
            lines.append(f"  <<: *{common_anchor}")
        lines.extend(("  " + line) if line.strip() else "" for line in body.splitlines())
        return "\n".join(lines) + "\n"


def _ordered_items(mapping, preferred):
    """把常用键排在前面,其余键保持原顺序,便于人工阅读导出的实验节。"""
    ordered = [(key, mapping[key]) for key in preferred if key in mapping]
    ordered += [(key, value) for key, value in mapping.items() if key not in preferred]
    return ordered


def _find_anchors(text):
    """找出顶层节定义的 anchor:``_common_experiment: &common_experiment`` -> 键名到 anchor 名。"""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^([A-Za-z0-9_.-]+)\s*:\s*&([^\s#]+)", text, re.MULTILINE)
    }


def _extract_argument_names(source_path):
    """从 argument_manager.py 提取 ``add_argument('--foo-bar')`` 定义的参数名。"""
    source = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    names.add(arg.value.lstrip("-").replace("-", "_"))
    if not names:
        raise McpToolError(f"未能从 {source_path} 解析出任何参数名")
    return names


def _section_line_range(text, name):
    """定位某个顶层节在文本里的行区间 [start, end)。

    遇到下一个顶格(非缩进、非空、非注释)的行即认为本节结束。
    """
    lines = text.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        if line.startswith(_ANCHOR_PREFIX) or line[:1].isspace() or not line.strip():
            continue
        if line.split(":", 1)[0].strip() == name:
            start = index
            break
    if start is None:
        raise McpToolError(f"在配置文本里定位不到实验节 {name!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            end = index
            break
    return sum(len(line) for line in lines[:start]), sum(len(line) for line in lines[:end])


def _atomic_write(path, text):
    """原子落盘:先写同目录临时文件再替换,避免中断留下半个配置。"""
    path = Path(path)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".exp_", suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _last_json_line(text):
    """取子进程输出里最后一行 JSON(前面可能混有警告等杂音)。"""
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                continue
    return None
