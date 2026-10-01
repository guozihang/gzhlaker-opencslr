# -*- coding: utf-8 -*-
"""运行时控制面单元测试(纯标准库,不依赖 torch)。

用 types.SimpleNamespace 和极小桩类构造假的 arg / optimizer / scheduler / model,
只验证 ``core/utils/runtime_control.py`` 的规则表与轮询/ack 逻辑。

用法(两种方式都可以):
    python3 core/tests/test_runtime_control.py
    python3 -m unittest discover -s core/tests -t . -p "test_runtime_control.py"
"""

import collections
import datetime
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from utils.runtime_control import (  # noqa: E402  需先补 sys.path
    HOT_KEYS,
    UNKNOWN_REASON,
    RuntimeControl,
    apply_overrides,
)


class FakeOptimizer:
    """最小优化器桩: 只需要 param_groups。"""

    def __init__(self, lrs, weight_decay=0.0):
        self.param_groups = [
            {"lr": lr, "weight_decay": weight_decay} for lr in lrs
        ]


class FakeScheduler:
    """最小 MultiStepLR 桩: base_lrs + milestones(Counter)。"""

    def __init__(self, base_lrs, milestones):
        self.base_lrs = list(base_lrs)
        self.milestones = collections.Counter(milestones)


class FakeLossModule:
    """持有 loss_weights 字典的假子模块。"""

    def __init__(self, loss_weights):
        self.loss_weights = loss_weights


class FakeModel:
    """最小模型桩: 只需要 modules()。"""

    def __init__(self, modules):
        self._modules = list(modules)

    def modules(self):
        return iter(self._modules)


def make_arg(**overrides):
    """构造一个与真实 argparse.Namespace 形状一致的假配置对象。"""
    data = {
        "optimizer_args": {"base_lr": 0.01, "weight_decay": 0.00005, "step": [5, 10]},
        "loss_weights": {"SeqCTC": 1.0},
        "num_epoch": 80,
        "save_interval": 200,
        "eval_interval": 100,
        "log_interval": 20,
        "print_log": True,
        "feeder_args": {},
    }
    data.update(overrides)
    return types.SimpleNamespace(**data)


def write_control(path, revision, overrides, issued_at="2024-01-01T00:00:00+00:00"):
    """按 MCP 侧约定写入控制文件。"""
    payload = {"revision": revision, "overrides": overrides, "issued_at": issued_at}
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False)


def read_ack(path):
    """读取 ack 文件。"""
    with open(path + ".ack.json", "r", encoding="utf-8") as reader:
        return json.load(reader)


class RuntimeControlTestCase(unittest.TestCase):
    """所有用例共享一个临时目录。"""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.control_path = os.path.join(self._tmpdir.name, "control.json")

    def build_control(self, log=None):
        """构造带假训练对象的 RuntimeControl。"""
        arg = make_arg()
        optimizer = FakeOptimizer([0.01, 0.01])
        scheduler = FakeScheduler([0.01, 0.01], [5, 10])
        loss_module = FakeLossModule(dict(arg.loss_weights))
        model = FakeModel([loss_module, object()])
        control = RuntimeControl(self.control_path, log=log)
        control.bind(arg=arg, optimizer=optimizer, scheduler=scheduler, model=model)
        return control, arg, optimizer, scheduler, loss_module, model


class TestDisabledAndMissingFile(RuntimeControlTestCase):
    """用例 1: 未启用 / 文件缺失时不报错、不写 ack。"""

    def test_none_path_is_noop(self):
        """control_file=None 时 poll 恒为 None, 且没有 ack 路径。"""
        control = RuntimeControl(None)
        self.assertIsNone(control.poll())
        self.assertIsNone(control.poll(epoch=0, batch=0))
        self.assertIsNone(control.ack_path)
        self.assertEqual(control.values["log_interval"], 200)

    def test_missing_control_file(self):
        """控制文件不存在时 poll 返回 None, 不抛异常, 不写 ack。"""
        warnings = []
        control, _, _, _, _, _ = self.build_control(log=warnings.append)
        self.assertIsNone(control.poll(epoch=0))
        self.assertFalse(os.path.exists(self.control_path + ".ack.json"))
        self.assertTrue(warnings)


class TestRevisionLifecycle(RuntimeControlTestCase):
    """用例 2/3: revision 幂等、ack 内容、新 revision 叠加。"""

    def test_first_revision_applies_and_ack_written(self):
        """revision 1 应用并写 ack; 同一 revision 不再重复应用。"""
        control, arg, optimizer, scheduler, _, _ = self.build_control()
        overrides = {"optimizer_args.base_lr": 0.005, "log_interval": 50}
        write_control(self.control_path, 1, overrides)

        self.assertEqual(control.poll(epoch=0, batch=0), overrides)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.005)
        self.assertEqual(arg.log_interval, 50)
        self.assertEqual(control.values["log_interval"], 50)

        ack = read_ack(self.control_path)
        self.assertEqual(ack["revision"], 1)
        self.assertIn("optimizer_args.base_lr", ack["applied"])
        self.assertEqual(ack["epoch"], 0)
        self.assertEqual(ack["batch"], 0)
        datetime.datetime.fromisoformat(ack["at"])  # at 必须是合法 ISO8601

        # 同一 revision 不再应用: 手工改回旧值后再次 poll 不应被覆盖
        optimizer.param_groups[0]["lr"] = 0.9
        scheduler.base_lrs = [0.9, 0.9]
        self.assertIsNone(control.poll(epoch=0, batch=20))
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.9)
        self.assertEqual(scheduler.base_lrs, [0.9, 0.9])

    def test_second_revision_applies_on_top(self):
        """revision 2 在 revision 1 基础上叠加, ack revision 匹配。"""
        control, arg, optimizer, _, _, _ = self.build_control()
        write_control(self.control_path, 1, {"num_epoch": 90})
        control.poll(epoch=0)
        self.assertEqual(arg.num_epoch, 90)

        write_control(self.control_path, 2, {"num_epoch": 120, "save_interval": 10})
        self.assertIsNotNone(control.poll(epoch=1))
        self.assertEqual(arg.num_epoch, 120)
        self.assertEqual(arg.save_interval, 10)
        ack = read_ack(self.control_path)
        self.assertEqual(ack["revision"], 2)
        self.assertEqual(ack["applied"]["save_interval"], 10)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)


class TestHotRules(RuntimeControlTestCase):
    """用例 4~7: 各热更新键的落地行为。"""

    def test_base_lr_updates_optimizer_and_scheduler(self):
        """base_lr 同时写 optimizer.param_groups[*]['lr'] 与 scheduler.base_lrs。"""
        arg = make_arg()
        optimizer = FakeOptimizer([0.01, 0.01, 0.01])
        scheduler = FakeScheduler([0.01, 0.01, 0.01], [5, 10])
        applied, ignored = apply_overrides(
            arg, optimizer=optimizer, scheduler=scheduler,
            overrides={"base_lr": 0.02},  # 简写键也要支持
        )
        self.assertEqual(ignored, {})
        self.assertEqual(applied, {"base_lr": 0.02})
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.02, 0.02, 0.02])
        self.assertEqual(scheduler.base_lrs, [0.02, 0.02, 0.02])
        self.assertEqual(arg.optimizer_args["base_lr"], 0.02)

    def test_weight_decay_updates_param_groups(self):
        """weight_decay 写入每个 param group。"""
        arg = make_arg()
        optimizer = FakeOptimizer([0.01], weight_decay=0.00005)
        applied, ignored = apply_overrides(
            arg, optimizer=optimizer,
            overrides={"optimizer_args.weight_decay": 0.001})
        self.assertEqual(ignored, {})
        self.assertEqual(applied["optimizer_args.weight_decay"], 0.001)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.001)
        self.assertEqual(arg.optimizer_args["weight_decay"], 0.001)

    def test_step_updates_scheduler_milestones_counter(self):
        """optimizer_args.step 把 milestones 重写为 collections.Counter。"""
        arg = make_arg()
        scheduler = FakeScheduler([0.01], [5, 10])
        applied, ignored = apply_overrides(
            arg, scheduler=scheduler,
            overrides={"optimizer_args.step": [3, 7, 12]})
        self.assertEqual(ignored, {})
        self.assertEqual(applied["optimizer_args.step"], [3, 7, 12])
        self.assertIsInstance(scheduler.milestones, collections.Counter)
        self.assertEqual(scheduler.milestones, collections.Counter([3, 7, 12]))
        self.assertEqual(scheduler.milestones[7], 1)

    def test_intervals_and_num_epoch_land_on_arg(self):
        """num_epoch / save_interval / eval_interval / log_interval 落到 arg 上。"""
        arg = make_arg()
        applied, ignored = apply_overrides(arg, overrides={
            "num_epoch": 100,
            "save_interval": 25,
            "eval_interval": 5,
            "log_interval": 50,
        })
        self.assertEqual(ignored, {})
        self.assertEqual(arg.num_epoch, 100)
        self.assertEqual(arg.save_interval, 25)
        self.assertEqual(arg.eval_interval, 5)
        self.assertEqual(arg.log_interval, 50)
        self.assertEqual(len(applied), 4)

    def test_feeder_args_and_print_log(self):
        """feeder_args 就地合并, print_log 直接赋值。"""
        arg = make_arg()
        feeder_args = arg.feeder_args
        applied, ignored = apply_overrides(arg, overrides={
            "feeder_args.max_eval_frames": 1200,
            "feeder_args.skip_failed_eval_batches": True,
            "print_log": False,
        })
        self.assertEqual(ignored, {})
        self.assertIs(arg.feeder_args, feeder_args)
        self.assertEqual(feeder_args["max_eval_frames"], 1200)
        self.assertIs(feeder_args["skip_failed_eval_batches"], True)
        self.assertIs(arg.print_log, False)
        self.assertEqual(len(applied), 3)

    def test_loss_weights_deep_merge_in_place(self):
        """loss_weights 就地深合并, 模型子模块的字典同步更新且引用不变。"""
        arg = make_arg()
        arg_ref = arg.loss_weights
        model = FakeModel([
            FakeLossModule(arg.loss_weights),
            FakeLossModule({"SeqCTC": 1.0}),
            object(),  # 没有 loss_weights 的子模块应被安全跳过
        ])
        applied, ignored = apply_overrides(arg, model=model, overrides={
            "loss_weights": {"SeqCTC": 0.5, "ConvCTC": 2.0}})
        self.assertEqual(ignored, {})
        # arg 上的原始引用仍然有效, 且已合并
        self.assertIs(arg.loss_weights, arg_ref)
        self.assertEqual(arg_ref, {"SeqCTC": 0.5, "ConvCTC": 2.0})
        # 两个持有 loss_weights 的子模块都被就地更新
        self.assertEqual(model._modules[0].loss_weights, {"SeqCTC": 0.5, "ConvCTC": 2.0})
        self.assertEqual(model._modules[1].loss_weights, {"SeqCTC": 0.5, "ConvCTC": 2.0})
        self.assertEqual(applied["loss_weights"]["ConvCTC"], 2.0)

    def test_loss_weights_without_any_dict_is_ignored(self):
        """既没有 arg.loss_weights 也没有模型字段时, 明确记为 ignored。"""
        arg = make_arg(loss_weights=["SeqCTC"])
        applied, ignored = apply_overrides(
            arg, model=FakeModel([object()]), overrides={"loss_weights": {"SeqCTC": 0.5}})
        self.assertEqual(applied, {})
        self.assertIn("loss_weights", ignored)

    def test_nested_override_dict_is_flattened(self):
        """嵌套字典写法 {"optimizer_args": {...}} 也要能用。"""
        arg = make_arg()
        optimizer = FakeOptimizer([0.01])
        scheduler = FakeScheduler([0.01], [5, 10])
        applied, ignored = apply_overrides(
            arg, optimizer=optimizer, scheduler=scheduler,
            overrides={"optimizer_args": {"base_lr": 0.004, "step": [2, 4]}})
        self.assertEqual(ignored, {})
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.004)
        self.assertEqual(scheduler.milestones, collections.Counter([2, 4]))
        self.assertIn("optimizer_args.base_lr", applied)


class TestIgnoredKeys(RuntimeControlTestCase):
    """用例 8/9: 非热更新键与错误类型都必须被忽略并给中文理由。"""

    def test_non_hot_keys_ignored_with_chinese_reason(self):
        """model_args / batch_size / random_seed / decode_mode 等 → ignored。"""
        control, arg, _, _, _, _ = self.build_control()
        write_control(self.control_path, 1, {
            "model_args": {"num_classes": 100},
            "batch_size": 32,
            "num_worker": 8,
            "random_seed": 123,
            "device": "1",
            "phase": "test",
            "work_dir": "/tmp/other",
            "decode_mode": "beam",
        })
        overrides = control.poll(epoch=3)
        self.assertIsNotNone(overrides)  # 有非法键时整体也不能抛异常
        ack = read_ack(self.control_path)
        self.assertEqual(ack["applied"], {})
        for key in ("batch_size", "num_worker", "random_seed", "device",
                    "phase", "work_dir", "decode_mode"):
            self.assertIn(key, ack["ignored"], key)
            self.assertTrue(ack["ignored"][key], key)
        # model_args 展开为点号键, 理由是"网络结构"
        self.assertIn("model_args.num_classes", ack["ignored"])
        self.assertIn("网络结构", ack["ignored"]["model_args.num_classes"])
        self.assertIn("DataLoader", ack["ignored"]["batch_size"])
        self.assertIn("启动前", ack["ignored"]["random_seed"])
        self.assertIn("解码器", ack["ignored"]["decode_mode"])
        # arg 未被改动
        self.assertEqual(arg.num_epoch, 80)

    def test_unknown_key_ignored(self):
        """未知/拼写错误的键 → 未知超参数。"""
        applied, ignored = apply_overrides(make_arg(), overrides={"base_lrr": 0.1})
        self.assertEqual(applied, {})
        self.assertEqual(ignored["base_lrr"], UNKNOWN_REASON)

    def test_wrong_types_ignored(self):
        """类型错误(字符串 base_lr、float num_epoch 等)被忽略且不抛异常。"""
        arg = make_arg()
        optimizer = FakeOptimizer([0.01])
        scheduler = FakeScheduler([0.01], [5, 10])
        applied, ignored = apply_overrides(
            arg, optimizer=optimizer, scheduler=scheduler,
            overrides={
                "base_lr": "oops",
                "num_epoch": 1.5,
                "log_interval": 0,
                "print_log": "yes",
                "optimizer_args.step": "5,10",
                "feeder_args.max_eval_frames": "many",
                "feeder_args.skip_failed_eval_batches": 1,
            })
        self.assertEqual(applied, {})
        self.assertEqual(len(ignored), 7)
        for reason in ignored.values():
            self.assertIn("类型错误", reason)
        # arg / optimizer 保持原样
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(arg.num_epoch, 80)

    def test_apply_overrides_handles_empty_and_bad_input(self):
        """overrides 为 None / 非字典时不抛异常。"""
        self.assertEqual(apply_overrides(make_arg()), ({}, {}))
        applied, ignored = apply_overrides(make_arg(), overrides=["not", "a", "dict"])
        self.assertEqual(applied, {})
        self.assertIn("overrides", ignored)


class TestInvalidInputs(RuntimeControlTestCase):
    """用例 10/11: 控制文件损坏/消失, 以及缺少 optimizer/scheduler 的行为。"""

    def test_malformed_json_returns_none(self):
        """控制文件内容不是合法 JSON → poll 返回 None, 不抛异常。"""
        warnings = []
        control, _, _, _, _, _ = self.build_control(log=warnings.append)
        with open(self.control_path, "w", encoding="utf-8") as writer:
            writer.write('{"revision": 1, "overrides": {')
        self.assertIsNone(control.poll(epoch=0))
        self.assertFalse(os.path.exists(self.control_path + ".ack.json"))
        self.assertTrue(warnings)

    def test_invalid_revision_and_non_dict_payload(self):
        """revision 非法或顶层不是对象时 → poll 返回 None。"""
        control, _, _, _, _, _ = self.build_control()
        with open(self.control_path, "w", encoding="utf-8") as writer:
            json.dump({"revision": 0, "overrides": {"num_epoch": 10}}, writer)
        self.assertIsNone(control.poll())
        with open(self.control_path, "w", encoding="utf-8") as writer:
            json.dump([1, 2, 3], writer)
        self.assertIsNone(control.poll())
        self.assertFalse(os.path.exists(self.control_path + ".ack.json"))

    def test_control_file_disappears_mid_run(self):
        """控制文件在运行中消失 → poll 返回 None, 不影响训练。"""
        control, arg, _, _, _, _ = self.build_control()
        write_control(self.control_path, 1, {"num_epoch": 90})
        self.assertIsNotNone(control.poll(epoch=0))
        self.assertEqual(arg.num_epoch, 90)
        os.remove(self.control_path)
        self.assertIsNone(control.poll(epoch=1))
        self.assertEqual(arg.num_epoch, 90)

    def test_base_lr_without_optimizer_is_ignored(self):
        """缺少 optimizer/scheduler 时 base_lr 不得谎报成功, 必须记为 ignored。"""
        arg = make_arg()
        applied, ignored = apply_overrides(
            arg, optimizer=None, scheduler=None, overrides={"base_lr": 0.02})
        self.assertEqual(applied, {})
        self.assertIn("base_lr", ignored)
        self.assertTrue(ignored["base_lr"])
        self.assertEqual(arg.optimizer_args["base_lr"], 0.01)

    def test_unwritable_ack_does_not_raise(self):
        """ack 写入失败(路径被目录占用)只告警, poll 仍返回 overrides。"""
        warnings = []
        control, arg, _, _, _, _ = self.build_control(log=warnings.append)
        os.mkdir(self.control_path + ".ack.json")  # 让 os.replace 失败
        write_control(self.control_path, 1, {"num_epoch": 70})
        overrides = control.poll(epoch=2)
        self.assertEqual(overrides, {"num_epoch": 70})
        self.assertEqual(arg.num_epoch, 70)
        self.assertTrue(warnings)


class TestValuesAndHotKeys(RuntimeControlTestCase):
    """values 暴露与 HOT_KEYS 规则表的一致性。"""

    def test_values_default_and_updates(self):
        """values 默认 log_interval=200; 应用后反映最新值。"""
        control, _, _, _, _, _ = self.build_control()
        self.assertEqual(control.values["log_interval"], 200)
        write_control(self.control_path, 1, {"log_interval": 33, "save_interval": 7})
        control.poll(epoch=0)
        self.assertEqual(control.values["log_interval"], 33)
        self.assertEqual(control.values["save_interval"], 7)

    def test_hot_keys_table_covers_rules(self):
        """HOT_KEYS 含点号全名, 且每个键都能被规则表识别(非未知)。"""
        self.assertIn("optimizer_args.base_lr", HOT_KEYS)
        self.assertIn("loss_weights", HOT_KEYS)
        arg = make_arg()
        optimizer = FakeOptimizer([0.01])
        scheduler = FakeScheduler([0.01], [5, 10])
        model = FakeModel([FakeLossModule({"SeqCTC": 1.0})])
        overrides = {
            "optimizer_args.base_lr": 0.02,
            "optimizer_args.weight_decay": 0.001,
            "optimizer_args.step": [1, 2],
            "loss_weights": {"SeqCTC": 0.5},
            "num_epoch": 10,
            "save_interval": 1,
            "eval_interval": 1,
            "log_interval": 1,
            "print_log": True,
            "feeder_args.max_eval_frames": 100,
            "feeder_args.skip_failed_eval_batches": False,
        }
        applied, ignored = apply_overrides(
            arg, optimizer=optimizer, scheduler=scheduler, model=model,
            overrides=overrides)
        self.assertEqual(ignored, {}, ignored)
        self.assertEqual(set(applied), set(overrides))


if __name__ == "__main__":
    unittest.main(verbosity=2)
