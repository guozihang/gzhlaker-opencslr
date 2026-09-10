# -*- coding: utf-8 -*-
"""样本统计 / 结果汇总集成测试(纯 CPU,无需 GPU 与真实数据集)。

覆盖 EXECUTION_PLAN.md §1.1 的三个集成点:
  1. ``DataloaderManager._worker_init`` 为每个 worker 派生确定性的
     numpy/random 种子(接入 utils.seed_utils.seed_worker);
  2. ``CUDAPrefetcher`` 在无 CUDA 时按原样透传 batch;
  3. ``seq_eval`` 记录成功/跳过/失败样本并落盘
     ``sample_statistics_{split}.json``,同时由 EvaluationManager 写出结构化的
     ``experiment_result_{split}.json``。

opencv / loguru / rich / wandb / pandas / yaml 缺失时自动注入最小桩模块,
因此没有完整训练环境的机器也能直接运行:

    cd core
    python tests/test_stats_integration.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util
import json
import random
import shutil
import sys
import tempfile
import types
from functools import partial
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[1]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))


def _install_stubs():
    """为缺失的可选依赖注入最小桩模块,使集成路径可在裸环境下导入。"""

    def is_missing(name):
        try:
            return importlib.util.find_spec(name) is None
        except (ImportError, ValueError):
            return True

    def noop(*args, **kwargs):
        return None

    if is_missing("cv2"):
        cv2 = types.ModuleType("cv2")
        cv2.COLOR_BGR2RGB = 4
        cv2.setNumThreads = noop
        cv2.imread = lambda *args, **kwargs: None
        cv2.cvtColor = lambda image, code: image
        sys.modules["cv2"] = cv2

    if is_missing("yaml"):
        yaml = types.ModuleType("yaml")
        yaml.FullLoader = object
        yaml.safe_load = lambda *args, **kwargs: {}
        yaml.load = lambda *args, **kwargs: {}
        sys.modules["yaml"] = yaml

    if is_missing("loguru"):
        loguru = types.ModuleType("loguru")
        loguru.logger = type("_Logger", (), {"__getattr__": lambda self, item: noop})()
        sys.modules["loguru"] = loguru

    if is_missing("wandb"):
        wandb = types.ModuleType("wandb")
        for name in ("init", "log", "warning", "error", "finish"):
            setattr(wandb, name, noop)
        sys.modules["wandb"] = wandb

    if is_missing("pandas"):
        pandas = types.ModuleType("pandas")
        pandas.DataFrame = type("_DataFrame", (), {})
        sys.modules["pandas"] = pandas

    if is_missing("rich"):
        rich = types.ModuleType("rich")

        class _Passthrough(object):
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs

        class _Console(_Passthrough):
            def print(self, *args, **kwargs):
                pass

        sys.modules["rich"] = rich
        for name, attr in (("console", "Console"), ("panel", "Panel"),
                           ("pretty", "Pretty"), ("rule", "Rule")):
            module = types.ModuleType("rich." + name)
            setattr(module, attr, _Console if name == "console" else _Passthrough)
            sys.modules["rich." + name] = module
            setattr(rich, name, module)


_install_stubs()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from manager.collect_manager import CollectManager  # noqa: E402
from manager.cuda_prefetcher import CUDAPrefetcher  # noqa: E402
from manager.dataloader_manager import DataloaderManager  # noqa: E402
from manager.device_manager import DeviceManager  # noqa: E402
from manager.evaluation_manager import EvaluationManager  # noqa: E402
from models.keys import Keys  # noqa: E402
from pipline.single import seq_eval  # noqa: E402
from utils.sample_statistics import SampleStatistics  # noqa: E402


class _FakeEvalDataset(object):
    """最小评估数据集:返回假视频张量、标签与 ``fileid|extra`` 形式的信息串。"""

    def __init__(self, sample_ids, num_gloss=3, num_frames=8):
        self.dict = {str(index): [index] for index in range(num_gloss)}
        self.sample_ids = list(sample_ids)
        self.num_frames = num_frames

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        video = torch.zeros(self.num_frames, 3, 16, 16)
        label = torch.LongTensor([1])
        return video, label, "{}|extra".format(self.sample_ids[index])


class _SeedProbeDataset(object):
    """记录 worker 内 numpy/random 首次取值,用于验证 worker 播种。"""

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return np.random.rand(), random.random()


class _FakeModel(torch.nn.Module):
    """伪模型:返回固定的识别结果,可选让首个 batch 抛 RuntimeError。"""

    def __init__(self, fail_first=False):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.fail_first = fail_first
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("synthetic eval failure")
        batch_size = data[Keys.VID].shape[0]
        return {
            Keys.LOSS: torch.zeros(()),
            Keys.TOTAL_LOSS: {"SeqCTC": 1.0},
            Keys.RECOGNIZED_SENTS: [[("word", 1.0)]] * batch_size,
        }


class _FakeCfg(object):
    """seq_eval 所需的最小配置对象(替代 argparse.Namespace)。"""

    def __init__(self, work_dir, **feeder_args):
        self.dataset = "phoenix2014"
        self.model = "slowfast"
        self.random_seed = 0
        self.config = "configs/exp.yaml"
        self.decode_mode = "greedy"
        self.dataset_info = {
            "evaluation_dir": "./libs/slr_eval",
            "evaluation_prefix": "phoenix2014-groundtruth",
        }
        self.feeder_args = feeder_args


def _fake_evaluate(**kwargs):
    """替换 EvaluationManager.evaluate,避免依赖 groundtruth STM 文件。"""
    return "Percent Total Error       =  23.45%   (OpenCSLR)"


def _make_loader(sample_ids, batch_size=2):
    return DataLoader(
        _FakeEvalDataset(sample_ids),
        batch_size=batch_size,
        collate_fn=CollectManager.collate,
    )


def test_worker_init_seeds_workers_deterministically():
    """同一 base seed 下 worker 的 numpy/random 序列可复现,不同 seed 则不同。"""
    previous_threads = torch.get_num_threads()
    try:
        torch.manual_seed(1234)
        DataloaderManager._worker_init(0, threads=1)
        first = (np.random.rand(), random.random())
        assert torch.get_num_threads() == 1, torch.get_num_threads()

        torch.manual_seed(1234)
        DataloaderManager._worker_init(0, threads=1)
        assert (np.random.rand(), random.random()) == first

        # DataLoader 在真实 worker 中会把 base_seed + worker_id 作为该 worker 的
        # initial_seed,这里手动模拟:不同 worker 应得到不同的增强随机流。
        torch.manual_seed(1235)
        DataloaderManager._worker_init(1, threads=1)
        assert (np.random.rand(), random.random()) != first
    finally:
        torch.set_num_threads(previous_threads)
    print("[OK] worker_init seeds numpy/random deterministically")


def test_worker_init_applied_by_dataloader():
    """真实 DataLoader worker 中,worker_init_fn 让两次迭代结果完全一致。"""
    loader = DataLoader(
        _SeedProbeDataset(),
        batch_size=1,
        num_workers=2,
        worker_init_fn=partial(DataloaderManager._worker_init, threads=1),
    )
    def drain():
        return [tuple(round(float(value), 12) for value in batch) for batch in loader]

    try:
        torch.manual_seed(1234)
        first = drain()
        torch.manual_seed(1234)
        second = drain()
    except Exception as err:  # 受限环境下无法 spawn worker 时跳过
        print(f"[SKIP] DataLoader workers unavailable: {type(err).__name__}: {err}")
        return
    assert first == second, (first, second)
    assert len(set(first)) > 1, "两个 worker 应产生不同的随机流"
    print("[OK] DataLoader workers seeded via worker_init_fn")


def test_cuda_prefetcher_falls_back_without_cuda():
    """无 CUDA 时 CUDAPrefetcher 直接透传 DataLoader 的 batch。"""
    loader = _make_loader(["s1", "s2", "s3"])
    prefetcher = CUDAPrefetcher(loader, "cpu")
    assert not prefetcher.enabled
    batches = list(prefetcher)
    assert len(batches) == 2, len(batches)
    video, video_length, labels, label_length, info = batches[0]
    assert video.shape[0] == 2 and video_length.shape[0] == 2
    assert len(info) == 2
    print("[OK] CUDAPrefetcher passthrough without CUDA")


def test_seq_eval_records_successes_and_results():
    """评估成功后记录全部样本,并写出统计与结构化结果文件。"""
    work_dir = tempfile.mkdtemp(prefix="opencslr_stats_") + os.sep
    try:
        cfg = _FakeCfg(work_dir, max_eval_frames=None, skip_failed_eval_batches=True)
        loader = _make_loader(["s1", "s2", "s3", "s4"])
        wer = seq_eval(cfg, loader, _FakeModel(), DeviceManager.output_device, "dev", 0, work_dir)
        assert wer == 23.45, wer

        stats = SampleStatistics.load(work_dir + "sample_statistics_dev.json")
        assert stats.total_samples == 4, stats.total_samples
        assert stats.num_successful == 4, stats.num_successful
        assert stats.num_skipped == 0 and stats.num_failed == 0
        assert stats.status == "valid", stats.status

        with open(work_dir + "experiment_result_dev.json", "r", encoding="utf-8") as reader:
            result = json.load(reader)
        assert result["wer"] == 23.45, result["wer"]
        assert result["split"] == "dev" and result["dataset"] == "phoenix2014"
        assert result["model"] == "slowfast" and result["decoder"] == "greedy"
        assert result["total_samples"] == 4 and result["successful_samples"] == 4
        assert result["status"] == "valid" and result["protocol"] == "unified"
        assert result["config_path"] == "configs/exp.yaml"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print("[OK] seq_eval records successful samples and writes result JSON")


def test_seq_eval_records_skipped_batches():
    """超过 max_eval_frames 的 batch 记为 skipped,跳过率超阈值时实验判为 invalid。"""
    work_dir = tempfile.mkdtemp(prefix="opencslr_stats_") + os.sep
    try:
        cfg = _FakeCfg(work_dir, max_eval_frames=2, skip_failed_eval_batches=True)
        loader = _make_loader(["s1", "s2", "s3", "s4"])
        seq_eval(cfg, loader, _FakeModel(), DeviceManager.output_device, "test", 0, work_dir)

        stats = SampleStatistics.load(work_dir + "sample_statistics_test.json")
        assert stats.num_skipped == 4 and stats.num_successful == 0
        assert stats.skip_reasons == {"frames_exceeded": ["s1", "s2", "s3", "s4"]}, stats.skip_reasons
        assert stats.status == "invalid", stats.status

        # test 分割的权威结果文件名为 experiment_result.json
        with open(work_dir + "experiment_result.json", "r", encoding="utf-8") as reader:
            result = json.load(reader)
        assert result["split"] == "test"
        assert result["skipped_samples"] == 4 and result["status"] == "invalid"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print("[OK] seq_eval records skipped batches and flags invalid experiments")


def test_seq_eval_records_failed_batches():
    """batch 前向失败且允许跳过时,该 batch 样本记为 failed,其余样本继续评估。"""
    work_dir = tempfile.mkdtemp(prefix="opencslr_stats_") + os.sep
    try:
        cfg = _FakeCfg(work_dir, max_eval_frames=None, skip_failed_eval_batches=True)
        loader = _make_loader(["s1", "s2", "s3", "s4"])
        wer = seq_eval(cfg, loader, _FakeModel(fail_first=True), DeviceManager.output_device, "dev", 0, work_dir)
        assert wer == 23.45, wer

        stats = SampleStatistics.load(work_dir + "sample_statistics_dev.json")
        assert stats.num_failed == 2, stats.num_failed
        assert stats.num_successful == 2, stats.num_successful
        assert stats.errors and stats.errors[0].error_type == "RuntimeError", stats.errors
        assert stats.num_successful + stats.num_failed == 4
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    print("[OK] seq_eval records failed batches and continues")


def main():
    DeviceManager.init("none")
    CollectManager.set_kernel_sizes(["K5", "P2"])
    assert DeviceManager.output_device.type == "cpu"

    original_evaluate = EvaluationManager.evaluate
    EvaluationManager.evaluate = staticmethod(_fake_evaluate)
    try:
        test_worker_init_seeds_workers_deterministically()
        test_worker_init_applied_by_dataloader()
        test_cuda_prefetcher_falls_back_without_cuda()
        test_seq_eval_records_successes_and_results()
        test_seq_eval_records_skipped_batches()
        test_seq_eval_records_failed_batches()
    finally:
        EvaluationManager.evaluate = original_evaluate
    print("\nAll stats integration tests passed.")


if __name__ == "__main__":
    main()
