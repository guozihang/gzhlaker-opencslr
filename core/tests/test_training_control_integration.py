# -*- coding: utf-8 -*-
"""训练循环 × 运行时控制面的集成测试(不需要 torch / GPU)。

``test_runtime_control.py`` 测的是规则表本身;这里测的是**接线**:
``ExperimentManager.run_train`` 的 while 循环、epoch 边界轮询、
``seq_train`` 的 batch 级轮询、ack 落盘,跑的都是仓库里的真实代码。

做法:把 torch / numpy / tqdm 以及几个 manager 边界换成最小桩(见
``_install_stubs``),真实的 ``pipeline.single.seq_train`` 与
``manager.experiment_manager.run_train`` 照常执行;评估与模型前向本身不是
被测对象,用假 loader / 假 model 顶替。

因为要往 sys.modules 里塞桩模块,直接在本进程里跑会污染同批用例,所以真正
执行的部分放在子进程里(``--driver``),父进程只做断言。

    python3 core/tests/test_training_control_integration.py
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1]
DRIVER_MARKER = "@@CONTROL_INTEGRATION@@"


# ====================================================================== 父进程


class TrainingLoopControlTests(unittest.TestCase):
    """一次子进程跑完两种场景,断言控制面对训练循环的实际影响。"""

    @classmethod
    def setUpClass(cls):
        cls.result = _run_driver_subprocess()

    def test_control_run_executes_the_real_loop(self):
        run = self.result["control_run"]
        self.assertEqual(run["stub_modules"], {}, "桩没装齐,可能导入了真实依赖")
        self.assertTrue(run["epochs_ran"] > 0, run)

    def test_num_epoch_change_extends_the_loop_mid_run(self):
        """起始 num_epoch=2,运行中被改成 4:while 循环必须每次重读。"""
        run = self.result["control_run"]
        self.assertEqual(run["initial_num_epoch"], 2)
        self.assertEqual(run["epochs_ran"], 4)

    def test_lr_change_lands_mid_epoch_not_only_at_the_boundary(self):
        """batch 循环内也要轮询:同一个 epoch 里学习率就该变。"""
        run = self.result["control_run"]
        lrs = run["learning_rates"]
        self.assertEqual(lrs[:20], [0.001] * 20, lrs[:25])
        self.assertEqual(set(lrs[20:]), {0.002}, lrs[20:])

    def test_log_interval_from_control_governs_the_print_cadence(self):
        run = self.result["control_run"]
        # 25 个 batch、interval=2 -> 每个 epoch 13 条;4 个 epoch 共 52 条
        self.assertEqual(run["batch_log_lines"], 52)
        self.assertEqual(run["epochs_ran"], 4)

    def test_ack_reports_what_applied_and_what_was_refused(self):
        run = self.result["control_run"]
        ack = run["ack"]
        self.assertEqual(ack["revision"], 2)
        self.assertIn(0.002, ack["applied"].values(), ack)
        # 单独跑一条 revision=1 的场景,验证「可以改的生效、不可以改的带理由被拒」
        first = run["first_ack"]
        self.assertEqual(first["revision"], 1)
        self.assertEqual(first["applied"].get("num_epoch"), 1, first)
        self.assertIn("batch_size", first["ignored"], first)
        self.assertIn("DataLoader", first["ignored"]["batch_size"])

    def test_without_control_file_behaviour_is_unchanged(self):
        run = self.result["plain_run"]
        self.assertEqual(run["epochs_ran"], 2, "没有控制文件时应按 num_epoch 跑完就停")
        self.assertEqual(set(run["learning_rates"]), {0.001}, "不应有任何热改")
        self.assertFalse(run["ack_exists"], "没有控制文件时不应写 ack")
        self.assertEqual(run["batch_log_lines"], 2, "默认间隔 200 -> 每 epoch 1 条")


# ====================================================================== 子进程


def _run_driver_subprocess():
    """在自己的子进程里执行驱动,避免桩模块污染同批用例。"""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--driver"],
        cwd=str(CORE_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(DRIVER_MARKER):
            return json.loads(line[len(DRIVER_MARKER):])
    raise AssertionError(
        "驱动没有输出结果。\nstdout:\n{}\nstderr:\n{}".format(proc.stdout[-2000:], proc.stderr[-2000:])
    )


def _install_stubs():
    """最小桩:torch / numpy / tqdm 以及若干 manager 边界。

    只替换「与被测逻辑无关的外部依赖」,``seq_train`` 与 ``run_train`` 本身
    仍是仓库里的真实实现。
    """
    import contextlib
    import types

    def module(name, **attrs):
        mod = sys.modules.get(name) or types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _FakeTensor(object):
        def __init__(self, value, device=None, dtype=None):
            self.value = value

        def item(self):
            return self.value

    class _Scaled(object):
        def __init__(self, loss):
            self.loss = loss

        def backward(self):
            return None

    class _GradScaler(object):
        def scale(self, loss):
            return _Scaled(loss)

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

    module("torch", tensor=_FakeTensor, save=lambda *a, **k: None)
    module("torch.nn")
    module("torch.distributed", ReduceOp=types.SimpleNamespace(MAX="max"),
           all_reduce=lambda *a, **k: None)
    module("torch.cuda", is_available=lambda: False)
    module("torch.cuda.amp",
           autocast=lambda *a, **k: contextlib.nullcontext(),
           GradScaler=_GradScaler)
    module("numpy",
           isinf=lambda value: value in (float("inf"), float("-inf")),
           isnan=lambda value: value != value,
           mean=lambda values: sum(values) / len(values) if values else 0.0)
    module("tqdm", tqdm=lambda iterable, **kwargs: iterable)

    class _LogManager(object):
        messages = []

        @classmethod
        def info(cls, message):
            cls.messages.append(str(message))

        @classmethod
        def error(cls, message):
            cls.messages.append("ERROR " + str(message))

    class _DeviceManager(object):
        output_device = "cpu"
        is_distributed = False
        rank = 0
        local_rank = 0
        gpu_list = []

        @classmethod
        def to(cls, value):
            return value

        @classmethod
        def is_main_process(cls):
            return True

        @classmethod
        def barrier(cls):
            return None

        @classmethod
        def cleanup(cls):
            return None

    class _DataloaderManager(object):
        DATALOADER = {}

        @classmethod
        def set_epoch(cls, name, epoch):
            return None

        @classmethod
        def get_iterator(cls, name):
            return cls.DATALOADER.get(name)

        @classmethod
        def shutdown(cls):
            return None

    class _Keys(object):
        VID = "vid"
        VID_LGT = "vid_lgt"
        LABEL = "label"
        LABEL_LGT = "label_lgt"
        LOSS = "loss"
        TOTAL_LOSS = "total_loss"

    module("manager.log_manager", LogManager=_LogManager)
    module("manager.device_manager", DeviceManager=_DeviceManager)
    module("manager.dataloader_manager", DataloaderManager=_DataloaderManager)
    module("manager.module_manager", ModuleManager=type("_ModuleManager", (), {}))
    module("manager.evaluation_manager", EvaluationManager=type("_EvaluationManager", (), {}))
    module("models", Keys=_Keys)
    module("utils.sample_statistics", SampleStatistics=type("_SampleStatistics", (), {}))
    module("utils.seed_utils",
           set_seed=lambda *a, **k: None,
           get_rng_state=lambda: {},
           set_rng_state=lambda *a, **k: None)


def _driver():
    """在桩环境里跑真实训练循环,返回两种场景的观测结果。"""
    import os
    import shutil
    import tempfile
    import types

    sys.path.insert(0, str(CORE_ROOT))
    _install_stubs()

    from manager import experiment_manager as experiment_module
    from manager.dataloader_manager import DataloaderManager
    from manager.log_manager import LogManager
    from utils.runtime_control import RuntimeControl

    ExperimentManager = experiment_module.ExperimentManager
    # 评估不是被测对象(需要真数据/真模型),只留一个假的 WER 出口
    experiment_module.seq_eval = lambda *args, **kwargs: 42.0

    class FakeOptimizer(object):
        def __init__(self):
            self.param_groups = [{"lr": 0.001, "weight_decay": 0.0}]

        def zero_grad(self):
            return None

        def step(self):
            return None

        def state_dict(self):
            return {}

    class FakeScheduler(object):
        def __init__(self):
            self.base_lrs = [0.001]
            self.milestones = {}

        def step(self):
            return None

        def state_dict(self):
            return {}

    class FakeLoss(object):
        def item(self):
            return 1.0

    def run_once(label, initial_num_epoch, with_control, batches=25, write_revision=None):
        work = tempfile.mkdtemp(prefix="ctl_integration_")
        try:
            control_path = os.path.join(work, "run.control.json") if with_control else None
            lr_seen = []
            forwards = {"count": 0}

            optimizer = FakeOptimizer()
            scheduler = FakeScheduler()

            if write_revision and control_path:
                # 起跑前先写一条:num_epoch 变 4、log_interval 变 2,外加一个
                # 不可热改的键,用来验证「拒绝也要给理由」
                _write_control(control_path, 1, write_revision["pre"])

            class FakeModel(object):
                def train(self):
                    return None

                def state_dict(self):
                    return {}

                def __call__(self, data):
                    lr_seen.append(optimizer.param_groups[0]["lr"])
                    forwards["count"] += 1
                    if write_revision and forwards["count"] == write_revision["at"]:
                        _write_control(control_path, 2, write_revision["mid"])
                    return {"loss": FakeLoss(), "total_loss": {"SeqCTC": 1.0}}

            arg = types.SimpleNamespace(
                optimizer_args={"base_lr": 0.001, "weight_decay": 0.0,
                                "step": [5, 10], "start_epoch": 0},
                num_epoch=initial_num_epoch,
                save_interval=1000,
                eval_interval=1000,
                log_interval=200,
                print_log=True,
                loss_weights={"SeqCTC": 1.0},
                random_fix=False,
                random_seed=0,
                work_dir=os.path.join(work, "wd"),
                control_file=control_path,
                phase="train",
                feeder_args={"mode": "train"},
            )

            # seq_train 的 _to_device 期待 (vid, vid_lgt, label, label_lgt) 四元组;
            # DeviceManager.to 在桩里是恒等函数,所以内容不重要
            DataloaderManager.DATALOADER = {
                "train": [("vid-%d" % index, index, [0], [1]) for index in range(batches)],
                # epoch 0 一定满足 epoch % eval_interval == 0,所以 dev/test 也会被取到
                # (真正的评估已被替换成假 seq_eval,这里只是让参数求值不报错)
                "dev": [],
                "test": [],
            }
            LogManager.messages = []

            ExperimentManager.arg = arg
            ExperimentManager.model = FakeModel()
            ExperimentManager.optimizer = optimizer
            ExperimentManager.scheduler = scheduler
            ExperimentManager.device = "cpu"

            ExperimentManager.run_train()

            ack_path = (control_path + ".ack.json") if control_path else None
            ack = _read_json(ack_path) if ack_path else None

            return {
                "label": label,
                "initial_num_epoch": initial_num_epoch,
                "epochs_ran": forwards["count"] // batches if batches else 0,
                "learning_rates": lr_seen,
                "batch_log_lines": sum(1 for m in LogManager.messages if "Batch(" in m),
                "ack": ack,
                "ack_exists": bool(ack_path and os.path.exists(ack_path)),
                "stub_modules": {},
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _write_control(path, revision, overrides):
        payload = {"revision": revision, "overrides": overrides, "issued_at": "2026-01-01T00:00:00+00:00"}
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)

    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, ValueError):
            return None

    # 场景一:控制面开着。revision 1 提前写好,revision 2 在第 5 次前向时写入,
    # 因此它最早会在 batch 20 的轮询点被应用 —— 同一个 epoch 内就该看到新学习率。
    control_run = run_once(
        "control",
        initial_num_epoch=2,
        with_control=True,
        write_revision={
            "pre": {"num_epoch": 4, "log_interval": 2, "batch_size": 8},
            "at": 5,
            "mid": {"optimizer_args": {"base_lr": 0.002}},
        },
    )
    # 场景二:不传 --control-file,行为必须与之前完全一致
    plain_run = run_once("plain", initial_num_epoch=2, with_control=False)

    # 补一个「第一条 revision 的 ack」:单独跑一次只会应用 revision 1 的场景
    single = run_once(
        "single",
        initial_num_epoch=1,
        with_control=True,
        write_revision={"pre": {"num_epoch": 1, "batch_size": 8}, "at": -1, "mid": {}},
    )
    control_run["first_ack"] = single["ack"]

    print(DRIVER_MARKER + json.dumps({"control_run": control_run, "plain_run": plain_run},
                                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if "--driver" in sys.argv:
        sys.exit(_driver())
    unittest.main(verbosity=2)
