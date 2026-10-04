# -*- encoding: utf-8 -*-
"""实验配置的读取、解析与创建(上游 immc-lab/OpenCSLR 布局)。

上游的配置模型与本仓库旧版(**分节的 exp.yaml + network.yaml + dataset.yaml**)
完全不同,这里全部按上游来:

- **一个实验 = ``core/configs/`` 下一个扁平 YAML**(``vac.yaml``、``tlp.yaml``、
  ``baseline.yaml``),顶层直接是参数,里面的 ``model: models.build_function.build_vac``
  是点号路径;
- ``core/configs/`` 里**同时**放着数据集配置(``phoenix2014.yaml`` 等,含
  ``dataset_root``/``dict_path``)。两者靠 ``model:`` 这个键区分:有 ``model:``
  的是实验,没有的是数据集;
- 上游 ``ConfigManager`` 只做 ``yaml.load`` + 字典合并,**没有嵌套键/取值校验**
  (连 ``base_lr`` 为负也照收)。所以这里的「校验」只覆盖能静态判定的部分:
  参数名是否在 argparse 里、``dataset`` 对应的数据集配置是否存在、
  ``model`` 点号路径能否在 ``core/`` 里定位到——取值域与嵌套键如实标为未知。

「实际生效的配置」仍然委托给 ``core_probe.py`` 跑上游真实的初始化链,不在这里复算。
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

# 实验名:YAML 文件名(不含扩展名)
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")

# 临时运行配置前缀。写在下划线开头:不会被当成实验节,也便于清理。
EPHEMERAL_PREFIX = "_mcp_run_"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# 数据集配置的判别键(上游 dataset_manager 读的就是这两个)
_DATASET_KEYS = ("dict_path", "dataset_root")

_PROBE_PATH = Path(__file__).resolve().with_name("core_probe.py")
_PROBE_TIMEOUT_SECONDS = 180


class ExperimentConfig:
    """上游配置目录的视图(每次调用都重新读盘,避免拿到过期配置)。"""

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
            raise McpToolError(f"{label} 顶层必须是映射(参数名 -> 值)")
        return data

    def config_path_for(self, name):
        """实验/数据集名 -> 配置文件路径。"""
        if not _NAME_PATTERN.match(str(name)):
            raise McpToolError(f"名字 {name!r} 不合法:只允许字母开头的字母/数字/._-")
        return self.layout.configs_dir / f"{name}.yaml"

    def experiment_doc(self, name):
        """某个实验配置文件的原始内容。"""
        path = self.config_path_for(name)
        if not path.is_file():
            raise McpToolError(
                f"实验 {name!r} 不存在({path});可用实验: {self.experiment_names()}"
            )
        doc = self._load_yaml(path, f"实验配置 {name}")
        if "model" not in doc:
            raise McpToolError(
                f"{path} 里没有 'model:' 键,它看起来是数据集配置而不是实验配置"
            )
        return doc

    def _all_config_docs(self):
        """configs/ 下所有非临时 YAML:名字 -> 内容。"""
        docs = {}
        if not self.layout.configs_dir.is_dir():
            return docs
        for path in sorted(self.layout.configs_dir.glob("*.yaml")):
            if path.stem.startswith(EPHEMERAL_PREFIX) or path.stem.startswith("_"):
                continue
            try:
                docs[path.stem] = self._load_yaml(path, f"配置 {path.stem}")
            except McpToolError:
                docs[path.stem] = None  # 坏文件也列出来,由 list_experiments 报问题
        return docs

    def experiment_names(self):
        """全部实验名 = configs/ 下带 ``model:`` 的 YAML。"""
        return sorted(
            name for name, doc in self._all_config_docs().items()
            if isinstance(doc, dict) and doc.get("model")
        )

    def dataset_names(self):
        """全部数据集名 = configs/ 下带 ``dataset_root``/``dict_path`` 的 YAML。"""
        names = []
        for name, doc in self._all_config_docs().items():
            if isinstance(doc, dict) and any(key in doc for key in _DATASET_KEYS):
                names.append(name)
        return sorted(names)

    def list_experiments(self):
        """实验总览:名称、模型点号路径、数据集、关键设置与明显问题。"""
        docs = self._all_config_docs()
        datasets = set(self.dataset_names())
        allowed = self.allowed_argument_names()
        items = []
        for name in sorted(docs):
            doc = docs[name]
            if not isinstance(doc, dict) or not doc.get("model"):
                continue
            problems = []
            model = doc.get("model")
            if not isinstance(model, str) or not model:
                problems.append("model 不是非空字符串")
            else:
                reason = self._model_reference_problem(model)
                if reason:
                    problems.append(reason)
            dataset = doc.get("dataset")
            if not dataset:
                problems.append("缺少 dataset 字段")
            elif dataset not in datasets:
                problems.append(f"数据集配置 configs/{dataset}.yaml 不存在")
            unknown = sorted(set(doc) - allowed)
            if unknown:
                problems.append(f"以下键不在 main.py 的参数表里,启动时会报错: {unknown}")
            items.append(
                {
                    "name": name,
                    "config_path": str(self.config_path_for(name)),
                    "model": model,
                    "dataset": dataset,
                    "phase": doc.get("phase"),
                    "work_dir": doc.get("work_dir"),
                    "device": doc.get("device"),
                    "num_epoch": doc.get("num_epoch"),
                    "problems": problems,
                }
            )
        return items

    def get_experiment_config(self, name):
        """某个实验**声明**了什么(配置文件原文 + 静态问题清单)。

        要看合并默认值之后「实际生效」的完整配置,用 ``resolve_experiment``。
        """
        doc = self.experiment_doc(name)
        notes = []
        model = doc.get("model")
        reason = self._model_reference_problem(model) if isinstance(model, str) else "model 不是字符串"
        if reason:
            notes.append(reason)
        dataset = doc.get("dataset")
        if dataset and dataset not in set(self.dataset_names()):
            notes.append(
                f"数据集配置 configs/{dataset}.yaml 不存在;"
                f"可用: {self.dataset_names()}"
            )
        unknown = sorted(set(doc) - self.allowed_argument_names())
        if unknown:
            notes.append(
                f"以下键不是 main.py 的参数,上游 map() 会断言失败: {unknown};"
                f"可用键见 main.py --help"
            )
        notes.append(
            "上游 ConfigManager 没有嵌套键/取值校验:model_args 等字典里的键"
            "写错不会在启动前被拦住"
        )
        return {
            "name": name,
            "config_path": str(self.config_path_for(name)),
            "config": doc,
            "notes": notes,
        }

    def list_options(self):
        """可选项:实验名、模型点号路径、数据集名。"""
        experiments = {item["name"]: item["model"] for item in self.list_experiments()}
        return {
            "experiments": sorted(experiments),
            "models": sorted({model for model in experiments.values() if model}),
            "datasets": self.dataset_names(),
            "configs_dir": str(self.layout.configs_dir),
        }

    def allowed_argument_names(self):
        """main.py 接受的参数名(蛇形),从 argument_manager.py 静态提取。

        直接解析源码而不是导入,是为了在没装 torch/toml 的机器上也能用;
        每次按文件 mtime 失效缓存,新增参数后无需重启 MCP 服务。
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

    def _model_reference_problem(self, dotted):
        """静态检查 ``models.build_function.build_vac`` 这种点号路径能否定位到。

        上游 ``ModuleManager.load`` 用 importlib 动态导入,写错要等到训练启动
        才炸。这里只做静态定位(文件 + 属性名),够拦住绝大多数拼写错误。
        """
        if not isinstance(dotted, str) or "." not in dotted:
            return f"model {dotted!r} 不是 '包.模块.函数' 形式的点号路径"
        module_path, attr = dotted.rsplit(".", 1)
        relative = Path(*module_path.split("."))
        for candidate in (self.layout.core_dir / relative.with_suffix(".py"),
                          self.layout.core_dir / relative / "__init__.py"):
            if candidate.is_file():
                try:
                    tree = ast.parse(candidate.read_text(encoding="utf-8"))
                except SyntaxError as exc:
                    return f"{candidate} 解析失败: {exc}"
                names = {node.name for node in tree.body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                names |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                          for target in node.targets if isinstance(target, ast.Name)}
                if attr not in names:
                    return f"{candidate} 里没有 {attr!r}"
                return None
        return f"model 指向的模块 {module_path!r} 在 core/ 下找不到({relative}.py)"

    # ------------------------------------------------------------------ 解析

    def resolve(self, name, overrides=None, timeout=_PROBE_TIMEOUT_SECONDS, config_path=None):
        """跑上游真实的初始化链,得到实际生效的配置。

        Args:
            name: 实验名(= configs/ 下的文件名)。
            overrides: 覆盖项;字典类按 YAML 语义深合并。
            timeout: 子进程超时(秒)。
            config_path: 要解析的配置文件;默认该实验自己的文件。生成临时运行
                配置后用它可以校验「这次真正要跑的东西」。
        """
        request = {
            "config": str(config_path or self.config_path_for(name)),
            "overrides": overrides or {},
        }
        return self._probe(request, name, timeout)

    def hyperparameters(self, name, config_path=None):
        """超参数清单:参数表 + 当前生效值 + 能否热改(上游没有热改)。"""
        request = {
            "mode": "schema",
            "config": str(config_path or self.config_path_for(name)),
        }
        return self._probe(request, name)

    def _probe(self, request, label, timeout=_PROBE_TIMEOUT_SECONDS):
        """把请求交给子进程里的上游初始化链,回传 JSON 响应。"""
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
            raise McpToolError(f"解析实验 {label!r} 超时(>{timeout}s)")

        response = _last_json_line(proc.stdout)
        if response is None:
            detail = (proc.stderr or proc.stdout or "").strip()[-800:]
            raise McpToolError(
                f"解析实验 {label!r} 的子进程没有返回结果(退出码 {proc.returncode}): {detail}"
            )
        return response

    # ------------------------------------------------------------------ 写入

    def create_experiment(self, name, model, dataset, overrides=None,
                          work_dir=None, device=None, phase=None, num_epoch=None,
                          overwrite=False):
        """在 configs/ 下新建一个实验配置文件(上游=一实验一文件)。

        Args:
            name: 实验名,同时是文件名(``<name>.yaml``)与客户端的引用名。
            model: 模型点号路径,如 ``models.build_function.build_vac``。
            dataset: 数据集名,必须在 configs/ 下有 ``<dataset>.yaml``。
            overrides: 其它参数(键同 main.py 参数名),可为嵌套字典。
            work_dir/device/phase/num_epoch: 常用参数的便捷入口。
            overwrite: 为 True 时覆盖同名文件。
        """
        path = self.config_path_for(name)
        existed = path.exists()
        if existed and not overwrite:
            raise McpToolError(f"实验 {name!r} 已存在({path});如需覆盖请传 overwrite=True")
        if not isinstance(model, str) or not model:
            raise McpToolError("model 必须是非空字符串(点号路径)")
        reason = self._model_reference_problem(model)
        if reason:
            raise McpToolError(f"model 无效: {reason}")
        if dataset not in set(self.dataset_names()):
            raise McpToolError(
                f"数据集 {dataset!r} 没有对应的 configs/{dataset}.yaml;"
                f"可用: {self.dataset_names()}"
            )

        fields = copy.deepcopy(overrides or {})
        if not isinstance(fields, dict):
            raise McpToolError("overrides 必须是映射(参数名 -> 值)")
        fields["model"] = model
        fields["dataset"] = dataset
        for key, value in (("work_dir", work_dir), ("device", device),
                           ("phase", phase), ("num_epoch", num_epoch)):
            if value is not None:
                fields[key] = value

        allowed = self.allowed_argument_names()
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise McpToolError(
                f"以下键不是 main.py 的参数: {unknown};可用键: {sorted(allowed)}"
            )

        ordered = _ordered_items(fields, ["model", "dataset", "work_dir", "device", "phase"])
        text = yaml.safe_dump(dict(ordered), allow_unicode=True,
                              default_flow_style=False, sort_keys=False, indent=2)
        _atomic_write(path, text)

        verdict = self.resolve(name)
        return {
            "name": name,
            "config_path": str(path),
            "replaced_existing": existed,
            "config": _load_yaml_text(text),
            "validation": {
                "ok": verdict.get("ok"),
                "error": verdict.get("error"),
                "error_type": verdict.get("error_type"),
            },
        }

    # -------------------------------------------------------------- 临时运行配置

    def ephemeral_config_path(self, run_id):
        """临时运行配置的路径(写在 configs/ 下,与实验配置同目录)。"""
        if not _RUN_ID_PATTERN.match(str(run_id)):
            raise McpToolError(f"非法的 run_id: {run_id!r}")
        return self.layout.configs_dir / f"{EPHEMERAL_PREFIX}{run_id}.yaml"

    def materialize_run_config(self, name, overrides=None, run_id=None):
        """把「某个实验 + 超参数覆盖」落成一份临时配置,不改动原文件。

        上游一个实验就是一个文件,所以「覆盖」= 复制该文件再把覆盖项深合并进去
        (字典类必须深合并:上游 ``map()`` 是 ``set_defaults``,整块替换会丢掉
        原文件里的兄弟键)。上游 ``ConfigManager`` 只加载你 ``--config`` 指向的
        那个文件,所以临时文件放哪都能跑;放在 configs/ 便于排查,名字以
        ``_mcp_run_`` 开头,不会被当成实验。

        Args:
            name: 实验名。
            overrides: 覆盖项,键同 main.py 参数名,可为嵌套字典。
            run_id: 运行 id,用于生成唯一文件名。

        Returns:
            dict: 临时配置路径、快照路径与生效覆盖项。
        """
        if not isinstance(overrides, dict):
            raise McpToolError("overrides 必须是映射(超参数名 -> 值)")
        if not run_id:
            raise McpToolError("生成临时运行配置需要 run_id")

        doc = self.experiment_doc(name)
        allowed = self.allowed_argument_names()
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise McpToolError(
                f"以下键不是 main.py 的参数: {unknown};可用键: {sorted(allowed)}"
            )

        merged = _deep_merge(doc, overrides)
        merged.setdefault("dataset", doc.get("dataset"))

        path = self.ephemeral_config_path(run_id)
        text = yaml.safe_dump(merged, allow_unicode=True,
                              default_flow_style=False, sort_keys=False, indent=2)
        _atomic_write(path, text)

        snapshot = self.layout.runs_dir / f"{run_id}.config.yaml"
        try:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(snapshot, text)
        except OSError:
            snapshot = None

        return {
            "run_id": run_id,
            "path": str(path),
            "snapshot": str(snapshot) if snapshot else None,
            "name": name,
            "model": merged.get("model"),
            "dataset": merged.get("dataset"),
            "overrides": overrides,
        }

    def cleanup_ephemeral_config(self, path):
        """删除一个临时运行配置(进程已结束,配置早就读完了)。"""
        if not path:
            return False
        candidate = Path(path)
        if candidate.parent != self.layout.configs_dir:
            return False
        if not candidate.name.startswith(EPHEMERAL_PREFIX):
            return False
        try:
            candidate.unlink()
            return True
        except OSError:
            return False

    def sweep_ephemeral_configs(self, active_run_ids=()):
        """清掉没有对应存活运行的临时配置(服务重启后的残留)。"""
        active = {str(item) for item in active_run_ids}
        removed = []
        if not self.layout.configs_dir.is_dir():
            return removed
        for path in sorted(self.layout.configs_dir.glob(f"{EPHEMERAL_PREFIX}*.yaml")):
            run_id = path.name[len(EPHEMERAL_PREFIX):-len(".yaml")]
            if run_id in active:
                continue
            if self.cleanup_ephemeral_config(path):
                removed.append(path.name)
        return removed


def _ordered_items(mapping, preferred):
    """把常用键排在前面,其余键保持原顺序,便于人工阅读导出的配置。"""
    ordered = [(key, mapping[key]) for key in preferred if key in mapping]
    ordered += [(key, value) for key, value in mapping.items() if key not in preferred]
    return ordered


def _deep_merge(base, override):
    """递归合并:嵌套字典逐层合并,其它值由 override 覆盖。"""
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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


def _atomic_write(path, text):
    """原子落盘:先写同目录临时文件再替换,避免中断留下半个配置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".cfg_", suffix=".yaml")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _load_yaml_text(text):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


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
