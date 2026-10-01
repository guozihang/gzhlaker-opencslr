# -*- encoding: utf-8 -*-
"""运行时控制面模块(纯标准库,不依赖 torch)。

长时训练进程在运行期间轮询一个 JSON 控制文件,把其中少量"可热更新"的
超参数(学习率、损失权重、epoch 数、各类 interval 等)应用到正在运行的
优化器 / 调度器 / 模型 / arg 上,并把结果写回 ack 文件供外部(MCP 服务)
读取确认。

外部写入的控制文件格式::

    {"revision": <int >= 1>,
     "overrides": {"<key>": <value>, ...},
     "issued_at": "<iso8601>"}

本进程写出的 ack 文件路径为 ``<control_file> + ".ack.json"``,内容::

    {"revision": <int>,
     "applied": {"<key>": <new value>},
     "ignored": {"<key>": "<reason>"},
     "at": "<iso8601>",
     "epoch": <int>, "batch": <int|null>}

设计约束:
  * 同一个 ``revision`` 不会被应用两次;
  * 控制面的任何错误(文件缺失、JSON 损坏、ack 不可写、未知键、类型错误)
    都只记录/忽略,绝不向训练循环抛异常;
  * 未提供 ``--control-file``(path 为 None)时一切退化为 no-op,
    训练行为与未接入该功能时完全一致;
  * 模块级只导入标准库,便于在没有 torch 的环境里单独测试规则表。
"""

import collections
import datetime
import importlib
import json
import os

__all__ = [
    "RuntimeControl",
    "apply_overrides",
    "HOT_KEYS",
    "HOT_KEY_TABLE",
    "IGNORED_REASONS",
    "UNKNOWN_REASON",
    "DEFAULT_VALUES",
]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 规则表:什么可以热更新,什么只能在启动前设置

# 可热更新的规范化键(``section.key`` 或顶层键)。
HOT_KEYS = (
    "optimizer_args.base_lr",
    "optimizer_args.weight_decay",
    "optimizer_args.step",
    "loss_weights",
    "num_epoch",
    "save_interval",
    "eval_interval",
    "log_interval",
    "print_log",
    "feeder_args.max_eval_frames",
    "feeder_args.skip_failed_eval_batches",
)

# 键 -> 中文说明,供文档/日志展示。
HOT_KEY_TABLE = {
    "optimizer_args.base_lr": "基础学习率; 同步 optimizer.param_groups[*]['lr'] 与 scheduler.base_lrs",
    "optimizer_args.weight_decay": "权重衰减; 写入每个 optimizer.param_groups 的 weight_decay",
    "optimizer_args.step": "MultiStepLR 的学习率衰减 epoch; 重写 scheduler.milestones(Counter)",
    "loss_weights": "损失权重; 就地深合并进 arg.loss_weights 与模型中持有 loss_weights 的子模块",
    "num_epoch": "总 epoch 数; epoch 循环每次重新读取,可中途延长或提前停止",
    "save_interval": "保存检查点的 epoch 间隔",
    "eval_interval": "评估的 epoch 间隔",
    "log_interval": "每多少个 batch 打印一次训练损失",
    "print_log": "是否打印日志",
    "feeder_args.max_eval_frames": "评估时允许的最大帧数; 下一次评估生效",
    "feeder_args.skip_failed_eval_batches": "评估是否跳过失败的 batch; 下一次评估生效",
}

# 非热更新键的中文拒绝理由。
_STARTUP_ONLY = "只能在启动前设置"
_DATALOADER_BOUND = "绑定 DataLoader, 需重启"
_MODEL_STRUCTURE = "属于网络结构, 只能在启动前修改"

IGNORED_REASONS = {
    "work_dir": _STARTUP_ONLY,
    "config": _STARTUP_ONLY,
    "exp": _STARTUP_ONLY,
    "phase": _STARTUP_ONLY,
    "device": _STARTUP_ONLY,
    "random_seed": _STARTUP_ONLY,
    "random_fix": _STARTUP_ONLY,
    "dataset": _STARTUP_ONLY,
    "dataset_info": _STARTUP_ONLY,
    "load_weights": _STARTUP_ONLY,
    "load_checkpoints": _STARTUP_ONLY,
    "ignore_weights": _STARTUP_ONLY,
    "feeder": _STARTUP_ONLY,
    "wandb": _STARTUP_ONLY,
    "model": _MODEL_STRUCTURE,
    "model_args": _MODEL_STRUCTURE,
    "batch_size": _DATALOADER_BOUND,
    "test_batch_size": _DATALOADER_BOUND,
    "num_worker": _DATALOADER_BOUND,
    "eval_num_worker": _DATALOADER_BOUND,
    "prefetch_factor": _DATALOADER_BOUND,
    "persistent_workers": _DATALOADER_BOUND,
    "pin_memory": _DATALOADER_BOUND,
    "worker_threads": _DATALOADER_BOUND,
    "preopen_memmap": _DATALOADER_BOUND,
    "gpu_prefetch": _DATALOADER_BOUND,
    "length_bucket_size": _DATALOADER_BOUND,
    "decode_mode": "在模型构建时已固化到解码器",
    "optimizer_args": _STARTUP_ONLY,
    "optimizer_args.optimizer": "更换优化器需重建优化器, 只能在启动前修改",
    "optimizer_args.start_epoch": _STARTUP_ONLY,
    "optimizer_args.learning_ratio": "在模型构建时已固化到各模块学习率",
    "optimizer_args.nesterov": "绑定优化器, 需重启",
    "feeder_args": _DATALOADER_BOUND,
}

# 未知(疑似拼写错误)键的拒绝理由。
UNKNOWN_REASON = "未知超参数"

# RuntimeControl.values 的初始值; log_interval 默认 200 以保持 seq_train 原有输出。
DEFAULT_VALUES = {"log_interval": 200}

# 允许省略 section 前缀的简写键 -> 规范化键。
_BARE_ALIASES = {
    "base_lr": "optimizer_args.base_lr",
    "weight_decay": "optimizer_args.weight_decay",
    "step": "optimizer_args.step",
    "optimizer": "optimizer_args.optimizer",
    "learning_ratio": "optimizer_args.learning_ratio",
    "start_epoch": "optimizer_args.start_epoch",
    "nesterov": "optimizer_args.nesterov",
    "max_eval_frames": "feeder_args.max_eval_frames",
    "skip_failed_eval_batches": "feeder_args.skip_failed_eval_batches",
    "mode": "feeder_args.mode",
}

# 允许以嵌套字典形式给出的配置节。
_NESTED_SECTIONS = ("model_args", "feeder_args", "optimizer_args")


def _is_number(value):
    """判断是否为数字(排除 bool,因为 bool 是 int 的子类)。"""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _is_int(value):
    """判断是否为整数(排除 bool)。"""
    return not isinstance(value, bool) and isinstance(value, int)


def _type_error(expected, value):
    """构造统一的中文类型错误理由。"""
    return "类型错误: 期望 {}, 实际 {}".format(expected, type(value).__name__)


def _ignored_reason(key):
    """为非热更新键给出中文拒绝理由。"""
    if key in IGNORED_REASONS:
        return IGNORED_REASONS[key]
    if key.startswith("model_args"):
        return _MODEL_STRUCTURE
    if key.startswith("feeder_args"):
        return _DATALOADER_BOUND
    if key.startswith("optimizer_args"):
        return _STARTUP_ONLY
    return UNKNOWN_REASON


def normalize_overrides(overrides):
    """把外部 ``overrides`` 归一化为 ``(display_key, canonical_key, value)`` 列表。

    ``display_key`` 是外部写入时的原始键名(ack 中沿用,便于 MCP 侧对齐),
    ``canonical_key`` 是规则表使用的规范化键。支持三种写法:

    * 点号全名: ``{"optimizer_args.base_lr": 0.1}``;
    * 简写: ``{"base_lr": 0.1}``;
    * 嵌套字典: ``{"optimizer_args": {"base_lr": 0.1}}``(展平为点号全名)。
    """
    normalized = []
    for key, value in overrides.items():
        if key in _NESTED_SECTIONS and isinstance(value, dict):
            for sub_key, sub_value in value.items():
                full = "{}.{}".format(key, sub_key)
                normalized.append((full, full, sub_value))
            continue
        canonical = _BARE_ALIASES.get(key, key)
        normalized.append((key, canonical, value))
    return normalized


def _deep_update(target, patch):
    """把 ``patch`` 就地深合并进 ``target``(嵌套字典递归,其余值直接覆盖)。"""
    for key, value in patch.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _assign_nested(arg, section, key, value):
    """若 ``arg.<section>`` 是字典,则就地写入 ``key``。"""
    container = getattr(arg, section, None)
    if isinstance(container, dict):
        container[key] = value


def apply_overrides(arg, optimizer=None, scheduler=None, model=None, overrides=None):
    """按固定规则表应用一组热更新覆盖值。

    纯函数式风格(只通过鸭子类型访问对象),不导入 torch,便于单元测试:

    * ``optimizer`` 需要 ``param_groups``(list[dict]);
    * ``scheduler`` 需要 ``base_lrs`` / ``milestones``(MultiStepLR);
    * ``model`` 需要 ``modules()``,用于查找持有 ``loss_weights`` 字典的子模块;
    * ``arg`` 是 argparse.Namespace,顶层键直接赋值,``optimizer_args`` /
      ``feeder_args`` / ``loss_weights`` 就地修改以保持既有引用有效。

    Args:
        arg: 配置对象(argparse.Namespace)。
        optimizer: 优化器; 学习率/权重衰减类覆盖值需要它。
        scheduler: MultiStepLR 调度器; base_lr / step 类覆盖值需要它。
        model: 模型; loss_weights 覆盖值会同步到子模块持有的字典。
        overrides: 外部写入的覆盖字典。

    Returns:
        tuple[dict, dict]: ``(applied, ignored)``;``applied`` 是键 -> 新值,
        ``ignored`` 是键 -> 中文拒绝/失败理由。任何单键失败都不会影响其它键,
        也不会抛异常。
    """
    applied = {}
    ignored = {}
    if not overrides:
        return applied, ignored
    if not isinstance(overrides, dict):
        ignored["overrides"] = "overrides 必须是 JSON 对象"
        return applied, ignored

    for display_key, canonical_key, value in normalize_overrides(overrides):
        try:
            _apply_one(arg, optimizer, scheduler, model,
                       display_key, canonical_key, value, applied, ignored)
        except Exception as err:  # 控制面错误绝不能冒泡到训练循环
            ignored[display_key] = "应用失败: {}".format(err)
    return applied, ignored


def _apply_one(arg, optimizer, scheduler, model,
               display_key, canonical_key, value, applied, ignored):
    """应用单个规范化键,结果写入 ``applied`` / ``ignored``。"""
    if canonical_key == "optimizer_args.base_lr":
        if not _is_number(value):
            ignored[display_key] = _type_error("number", value)
        elif value <= 0:
            ignored[display_key] = "base_lr 必须为正数, 实际 {}".format(value)
        elif optimizer is None or not getattr(optimizer, "param_groups", None):
            # optimizer/scheduler 缺失时不能假装成功,明确记为 ignored
            ignored[display_key] = "优化器不可用, 无法热更新学习率"
        else:
            groups = list(optimizer.param_groups)
            for group in groups:
                group["lr"] = value
            # MultiStepLR 的 get_lr 以 base_lrs 为基准,必须同步,否则下次 step() 会覆盖回旧值
            if scheduler is not None and hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [value] * len(groups)
            _assign_nested(arg, "optimizer_args", "base_lr", value)
            applied[display_key] = value
        return

    if canonical_key == "optimizer_args.weight_decay":
        if not _is_number(value):
            ignored[display_key] = _type_error("number", value)
        elif optimizer is None or not getattr(optimizer, "param_groups", None):
            ignored[display_key] = "优化器不可用, 无法热更新权重衰减"
        else:
            for group in optimizer.param_groups:
                group["weight_decay"] = value
            _assign_nested(arg, "optimizer_args", "weight_decay", value)
            applied[display_key] = value
        return

    if canonical_key == "optimizer_args.step":
        if not isinstance(value, list) or not all(_is_int(item) for item in value):
            ignored[display_key] = _type_error("list[int]", value)
        elif scheduler is None or not hasattr(scheduler, "milestones"):
            ignored[display_key] = "学习率调度器不可用, 无法热更新 milestones"
        else:
            # MultiStepLR 内部把 milestones 存为 collections.Counter
            scheduler.milestones = collections.Counter(value)
            _assign_nested(arg, "optimizer_args", "step", list(value))
            applied[display_key] = list(value)
        return

    if canonical_key == "loss_weights":
        if not isinstance(value, dict):
            ignored[display_key] = _type_error("dict", value)
            return
        bad_keys = [key for key, weight in value.items() if not _is_number(weight)]
        if bad_keys:
            ignored[display_key] = "类型错误: 损失权重必须为 number, 非法键 {}".format(bad_keys)
            return
        updated = 0
        effective = None
        # arg.loss_weights 会被按引用传给损失模块,就地更新即可让训练立刻生效
        target = getattr(arg, "loss_weights", None) if arg is not None else None
        if isinstance(target, dict):
            _deep_update(target, value)
            updated += 1
            effective = target
        # 模型可能持有同一字典的副本,遍历子模块逐个就地更新,保持既有引用有效
        if model is not None and hasattr(model, "modules"):
            for module in model.modules():
                module_weights = getattr(module, "loss_weights", None)
                if isinstance(module_weights, dict) and module_weights is not target:
                    _deep_update(module_weights, value)
                    updated += 1
        if updated == 0:
            ignored[display_key] = "未找到可更新的 loss_weights 字典"
        else:
            applied[display_key] = dict(effective) if isinstance(effective, dict) else dict(value)
        return

    if canonical_key == "num_epoch":
        if not _is_int(value) or value < 1:
            ignored[display_key] = _type_error("int >= 1", value)
        elif arg is None:
            ignored[display_key] = "配置对象(arg)不可用"
        else:
            arg.num_epoch = value
            applied[display_key] = value
        return

    if canonical_key in ("save_interval", "eval_interval", "log_interval"):
        if not _is_int(value) or value < 1:
            ignored[display_key] = _type_error("int >= 1", value)
        elif arg is None:
            ignored[display_key] = "配置对象(arg)不可用"
        else:
            setattr(arg, canonical_key, value)
            applied[display_key] = value
        return

    if canonical_key == "print_log":
        if not isinstance(value, bool):
            ignored[display_key] = _type_error("bool", value)
        elif arg is None:
            ignored[display_key] = "配置对象(arg)不可用"
        else:
            arg.print_log = value
            applied[display_key] = value
        return

    if canonical_key == "feeder_args.max_eval_frames":
        if value is not None and (not _is_int(value) or value < 1):
            ignored[display_key] = _type_error("int >= 1 或 null", value)
        elif arg is None or not isinstance(getattr(arg, "feeder_args", None), dict):
            ignored[display_key] = "feeder_args 不可用, 无法热更新"
        else:
            arg.feeder_args["max_eval_frames"] = value
            applied[display_key] = value
        return

    if canonical_key == "feeder_args.skip_failed_eval_batches":
        if not isinstance(value, bool):
            ignored[display_key] = _type_error("bool", value)
        elif arg is None or not isinstance(getattr(arg, "feeder_args", None), dict):
            ignored[display_key] = "feeder_args 不可用, 无法热更新"
        else:
            arg.feeder_args["skip_failed_eval_batches"] = value
            applied[display_key] = value
        return

    # 非热更新键 / 未知键:一律忽略,并给出中文理由
    ignored[display_key] = _ignored_reason(canonical_key)


def _utc_now_iso():
    """当前 UTC 时间的 ISO8601 字符串(带时区)。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _is_main_process():
    """是否主进程; 无法导入 DeviceManager(例如环境无 torch)时保守认为 True。

    这里用 importlib 延迟导入: ``manager.device_manager`` 依赖 torch,
    不能在模块级导入(否则纯标准库测试会失败),所以本模块的顶层导入保持纯标准库。
    """
    try:
        module = importlib.import_module("manager.device_manager")
        return bool(module.DeviceManager.is_main_process())
    except Exception:
        return True


class RuntimeControl:
    """运行时控制面:轮询控制文件、应用热更新并写回 ack。

    ``path`` 为 None 时全部退化为 no-op,不读文件、不写 ack。
    典型用法(ExperimentManager)::

        control = RuntimeControl(getattr(arg, "control_file", None))
        control.bind(arg=arg, optimizer=optimizer, scheduler=scheduler, model=model)
        ...
        control.poll(epoch=epoch)          # epoch 边界
        control.poll(epoch=epoch, batch=i) # batch 循环内周期性轮询
    """

    def __init__(self, path, log=None, arg=None, optimizer=None, scheduler=None, model=None):
        """初始化。

        Args:
            path: 控制文件路径; None 表示关闭运行时控制(no-op)。
            log: 可选的日志出口; 可以是 callable,也可以是带 info/warning 方法的对象。
            arg: 可选,配置对象(argparse.Namespace); 也可稍后通过 bind() 绑定。
            optimizer: 可选,优化器。
            scheduler: 可选,MultiStepLR 调度器。
            model: 可选,模型。
        """
        self.path = os.fspath(path) if path else None
        self.ack_path = self.path + ".ack.json" if self.path else None
        self.log = log
        self.arg = arg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self._last_revision = 0
        self._values = dict(DEFAULT_VALUES)
        self._warned = set()
        # 最近一次应用的结果,供调用方打日志
        self.last_revision = None
        self.last_applied = {}
        self.last_ignored = {}

    def bind(self, arg=None, optimizer=None, scheduler=None, model=None):
        """绑定训练对象(只覆盖显式传入的非 None 值)。"""
        if arg is not None:
            self.arg = arg
        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler
        if model is not None:
            self.model = model

    @property
    def values(self):
        """当前生效的热更新值(只读使用),例如 ``values["log_interval"]``。"""
        return self._values

    def poll(self, epoch=None, batch=None):
        """轮询一次控制文件; 有新 revision 时应用并写 ack。

        Args:
            epoch: 当前 epoch,写入 ack。
            batch: 当前 batch(epoch 边界轮询时为 None),写入 ack。

        Returns:
            dict | None: 新 revision 的 ``overrides`` 字典; 无新版本/出错/未启用时返回 None。
            被忽略的键与理由见 ``self.last_applied`` / ``self.last_ignored``。
        """
        if self.path is None:
            return None

        payload = self._read_payload()
        if payload is None:
            return None

        revision = payload.get("revision")
        if not _is_int(revision) or revision < 1:
            self._warn_once("控制文件 revision 非法(需为 >= 1 的整数), 已忽略")
            return None
        if revision <= self._last_revision:
            return None

        overrides = payload.get("overrides")
        malformed = not isinstance(overrides, dict)
        if malformed:
            self._warn_once("控制文件 overrides 必须是 JSON 对象, 已按空覆盖处理")
            overrides = {}

        if self.arg is None:
            applied = {}
            ignored = {key: "运行时控制未绑定配置对象(arg)" for key in overrides}
        else:
            try:
                applied, ignored = apply_overrides(
                    self.arg,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    model=self.model,
                    overrides=overrides,
                )
            except Exception as err:  # 兜底,保证训练循环不受影响
                applied = {}
                ignored = {key: "应用失败: {}".format(err) for key in overrides}
        if malformed:
            ignored["overrides"] = "overrides 必须是 JSON 对象"

        # 同一个 revision 绝不重复应用:无论成败都推进水位
        self._last_revision = revision
        self.last_revision = revision
        self.last_applied = dict(applied)
        self.last_ignored = dict(ignored)
        self._update_values(applied)
        self._write_ack(revision, applied, ignored, epoch, batch)
        self._emit_summary(revision, applied, ignored)
        return overrides

    def _read_payload(self):
        """读取并解析控制文件; 失败时返回 None(记录一次告警)。"""
        try:
            with open(self.path, "r", encoding="utf-8") as reader:
                payload = json.load(reader)
        except FileNotFoundError:
            self._warn_once("控制文件不存在, 跳过本次轮询")
            return None
        except (OSError, ValueError) as err:
            self._warn_once("控制文件读取失败({}): {}".format(type(err).__name__, err))
            return None
        if not isinstance(payload, dict):
            self._warn_once("控制文件内容必须是 JSON 对象, 已忽略")
            return None
        return payload

    def _update_values(self, applied):
        """把已应用的键写入 values(ack 中的原始键名先归一化)。"""
        for key, value in applied.items():
            canonical = _BARE_ALIASES.get(key, key)
            if canonical in HOT_KEYS:
                self._values[canonical] = value

    def _write_ack(self, revision, applied, ignored, epoch, batch):
        """写 ack 文件(仅主进程); 失败只告警,不抛异常。"""
        if self.ack_path is None or not _is_main_process():
            return
        payload = {
            "revision": revision,
            "applied": dict(applied),
            "ignored": dict(ignored),
            "at": _utc_now_iso(),
            "epoch": epoch,
            "batch": batch,
        }
        tmp_path = self.ack_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as writer:
                json.dump(payload, writer, ensure_ascii=False)
            os.replace(tmp_path, self.ack_path)
        except Exception as err:
            self._warn_once("写入 ack 文件失败: {}".format(err))

    def _emit_summary(self, revision, applied, ignored):
        """应用成功后输出一条中文摘要(含 applied/ignored 键)。"""
        if not applied and not ignored:
            return
        self._emit("运行时控制 revision {} 已处理: applied={}, ignored={}".format(
            revision, sorted(applied), sorted(ignored)))

    def _emit(self, message, level="info"):
        """安全地输出日志(log 可以是 callable 或日志对象),绝不抛异常。"""
        if self.log is None:
            return
        try:
            if callable(self.log):
                self.log(message)
                return
            handler = getattr(self.log, level, None) or getattr(self.log, "info", None)
            if handler is not None:
                handler(message)
        except Exception:
            pass

    def _warn_once(self, message):
        """同一个告警只输出一次,避免每个 batch 刷屏。"""
        if message in self._warned:
            return
        self._warned.add(message)
        self._emit(message, level="warning")
