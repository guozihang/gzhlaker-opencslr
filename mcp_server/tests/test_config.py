# -*- encoding: utf-8 -*-
"""实验配置的读取、创建与临时运行配置。

上游约定:**一个实验 = ``core/configs/`` 下一个扁平 YAML**(``vac.yaml``),
``model: models.build_function.build_vac`` 是点号路径;同一个目录里还放着数据集
配置(``phoenix2014.yaml`` 等),靠 ``model:`` / ``dataset_root`` 区分。
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import yaml

from mcp_server.config import EPHEMERAL_PREFIX, ExperimentConfig
from mcp_server.errors import McpToolError
from mcp_server.paths import RepoLayout

from .helpers import REPO_ROOT, env_without_root_override, make_temp_repo, remove_tree


class RealRepoReadTests(unittest.TestCase):
    """针对真实仓库 configs/ 的只读断言(上游配置本身也要经得起分类)。"""

    @classmethod
    def setUpClass(cls):
        cls.layout = RepoLayout(REPO_ROOT)
        cls.config = ExperimentConfig(cls.layout)

    def test_experiments_and_datasets_are_classified_by_their_keys(self):
        experiments = self.config.experiment_names()
        datasets = self.config.dataset_names()

        for name in ("baseline", "tlp", "vac"):
            self.assertIn(name, experiments)
        for name in ("phoenix2014", "phoenix2014-T", "CSL", "CSL-Daily"):
            self.assertIn(name, datasets)
        # 有 dataset_root/dict_path 的是数据集,不能混进实验
        for name in datasets:
            self.assertNotIn(name, experiments)
        # 有 model: 的是实验,不能被当成数据集
        for name in experiments:
            self.assertNotIn(name, datasets)
        # 临时运行配置(下划线开头)永远不算实验
        self.assertFalse([name for name in experiments if name.startswith("_")])

    def test_list_experiments_reports_dotted_model_path(self):
        items = {item["name"]: item for item in self.config.list_experiments()}
        self.assertEqual(set(items), set(self.config.experiment_names()))

        vac = items["vac"]
        self.assertEqual(vac["model"], "models.build_function.build_vac")
        self.assertEqual(vac["dataset"], "phoenix2014")
        self.assertEqual(vac["config_path"], str(self.layout.configs_dir / "vac.yaml"))
        self.assertEqual(vac["num_epoch"], 80)
        self.assertEqual(vac["problems"], [], "真实配置不应该有任何静态问题")

    def test_allowed_argument_names_come_from_the_upstream_parser(self):
        names = self.config.allowed_argument_names()
        for expected in ("config", "work_dir", "phase", "device", "dataset",
                         "model", "model_args", "decode_mode", "num_epoch"):
            self.assertIn(expected, names)
        # 旧布局的分节参数在上游的 parser 里已经不存在
        self.assertNotIn("exp", names)
        self.assertNotIn("network", names)
        self.assertNotIn("not_an_argument", names)

    def test_get_experiment_config_returns_the_raw_document(self):
        payload = self.config.get_experiment_config("vac")
        self.assertEqual(payload["name"], "vac")
        self.assertEqual(payload["config_path"], str(self.layout.configs_dir / "vac.yaml"))
        self.assertEqual(payload["config"]["model"], "models.build_function.build_vac")
        self.assertEqual(payload["config"]["dataset"], "phoenix2014")
        # 必须如实说明上游没有嵌套键/取值校验,不能假装校验过
        self.assertTrue(any("没有嵌套键/取值校验" in note for note in payload["notes"]))
        self.assertFalse(any("不存在" in note for note in payload["notes"]), payload["notes"])

    def test_dataset_config_is_not_an_experiment(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.get_experiment_config("phoenix2014")
        self.assertIn("model:", str(ctx.exception))

    def test_unknown_experiment_lists_alternatives(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.get_experiment_config("not-an-experiment")
        message = str(ctx.exception)
        self.assertIn("not-an-experiment", message)
        self.assertIn("vac", message)

    def test_list_options(self):
        options = self.config.list_options()
        self.assertEqual(options["experiments"], self.config.experiment_names())
        self.assertIn("models.build_function.build_vac", options["models"])
        self.assertNotIn("networks", options, "上游没有 network 节,不该再暴露这个概念")
        self.assertEqual(options["datasets"], self.config.dataset_names())
        self.assertEqual(options["configs_dir"], str(self.layout.configs_dir))

    def test_illegal_config_name_is_rejected(self):
        for name in ("_private", "1abc", "with space", "../escape"):
            with self.subTest(name=name):
                with self.assertRaises(McpToolError):
                    self.config.config_path_for(name)


class ResolveTests(unittest.TestCase):
    """resolve() 跑的是真实探针子进程,结论必须与启动训练时一致。"""

    @classmethod
    def setUpClass(cls):
        cls.layout = RepoLayout(REPO_ROOT)
        cls.config = ExperimentConfig(cls.layout)

    def test_without_overrides_reflects_the_config_file(self):
        verdict = self.config.resolve("vac")
        self.assertTrue(verdict["ok"], verdict)
        effective = verdict["effective"]
        self.assertEqual(effective["model"], "models.build_function.build_vac")
        self.assertEqual(effective["dataset"], "phoenix2014")
        self.assertEqual(effective["decode_mode"], "beam")
        self.assertEqual(effective["num_epoch"], 80)
        # map() 会把 configs/<dataset>.yaml 读进 dataset_info(cwd 必须是 core/)
        dataset_doc = yaml.safe_load(
            (self.layout.configs_dir / "phoenix2014.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(effective["dataset_info"]["dict_path"], dataset_doc["dict_path"])

    def test_overrides_are_applied_with_dicts_deep_merged(self):
        verdict = self.config.resolve("vac", {"num_epoch": 3, "model_args": {"use_bn": 0}})
        self.assertTrue(verdict["ok"], verdict)
        effective = verdict["effective"]
        self.assertEqual(effective["num_epoch"], 3)
        self.assertEqual(effective["model_args"]["use_bn"], 0)

        declared = yaml.safe_load(
            (self.layout.configs_dir / "vac.yaml").read_text(encoding="utf-8")
        )["model_args"]
        for key in ("num_classes", "c2d_type", "kernel_size", "stride"):
            self.assertEqual(effective["model_args"][key], declared[key], f"{key} 被丢掉了")

    def test_unknown_override_key_is_reported_not_raised(self):
        verdict = self.config.resolve("vac", {"definitely_not_a_key": 1})
        self.assertFalse(verdict["ok"], verdict)
        self.assertIn("definitely_not_a_key", verdict["error"])

    def test_invalid_value_reports_the_argparse_message(self):
        verdict = self.config.resolve("vac", {"num_epoch": "abc"})
        self.assertFalse(verdict["ok"], verdict)
        self.assertEqual(verdict["error_type"], "ValueError")
        self.assertIn("--num-epoch", verdict["error"])
        self.assertIn("abc", verdict["error"])


class CreateTests(unittest.TestCase):
    """在临时仓库里验证写入行为(不碰真实 core/)。"""

    def setUp(self):
        self.layout = make_temp_repo()
        self.config = ExperimentConfig(self.layout)
        self.experiment_file = self.layout.configs_dir / "vac.yaml"
        self.original_text = self.experiment_file.read_text(encoding="utf-8")

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_create_writes_a_flat_experiment_config(self):
        result = self.config.create_experiment(
            name="vac_tune",
            model="models.build_function.build_vac",
            dataset="phoenix2014",
            overrides={"batch_size": 4},
            work_dir="./work_dir/vac_tune/",
            device="1",
            phase="train",
            num_epoch=12,
        )
        path = self.layout.configs_dir / "vac_tune.yaml"
        self.assertEqual(Path(result["config_path"]), path)
        self.assertFalse(result["replaced_existing"])
        self.assertTrue(result["validation"]["ok"], result["validation"])

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["model"], "models.build_function.build_vac")
        self.assertEqual(doc["dataset"], "phoenix2014")
        self.assertEqual(doc["batch_size"], 4)
        self.assertEqual(doc["work_dir"], "./work_dir/vac_tune/")
        self.assertEqual(doc["device"], "1")
        self.assertEqual(doc["phase"], "train")
        self.assertEqual(doc["num_epoch"], 12)
        # 一实验一文件:原实验的文件一个字节都不该动
        self.assertEqual(self.experiment_file.read_text(encoding="utf-8"), self.original_text)
        self.assertIn("vac_tune", self.config.experiment_names())

    def test_created_experiment_passes_the_real_config_chain(self):
        self.config.create_experiment(
            name="vac_check",
            model="models.build_function.build_vac",
            dataset="phoenix2014",
        )
        verdict = self.config.resolve("vac_check")
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["effective"]["model"], "models.build_function.build_vac")

    def test_bogus_model_dotted_path_is_rejected(self):
        cases = (
            ("models.build_function.build_nope", "build_nope"),
            ("models.no_such_module.build_vac", "no_such_module"),
            ("not_a_dotted_path", "点号路径"),
        )
        for model, keyword in cases:
            with self.subTest(model=model):
                with self.assertRaises(McpToolError) as ctx:
                    self.config.create_experiment(
                        name="bad_model", model=model, dataset="phoenix2014"
                    )
                self.assertIn(keyword, str(ctx.exception))
                self.assertFalse((self.layout.configs_dir / "bad_model.yaml").exists())

    def test_missing_dataset_config_is_rejected(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(
                name="x", model="models.build_function.build_vac", dataset="CSL"
            )
        message = str(ctx.exception)
        self.assertIn("CSL", message)
        self.assertIn("phoenix2014", message, "报错要列出可用的数据集")
        self.assertFalse((self.layout.configs_dir / "x.yaml").exists())

    def test_unknown_override_key_is_rejected_without_writing(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(
                name="bad_key",
                model="models.build_function.build_vac",
                dataset="phoenix2014",
                overrides={"num_epochs": 3},
            )
        self.assertIn("num_epochs", str(ctx.exception))
        self.assertFalse((self.layout.configs_dir / "bad_key.yaml").exists())
        self.assertEqual(self.experiment_file.read_text(encoding="utf-8"), self.original_text)

    def test_duplicate_name_requires_overwrite(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(
                name="vac", model="models.build_function.build_tlp", dataset="CSL-Daily"
            )
        self.assertIn("overwrite", str(ctx.exception))
        self.assertEqual(self.experiment_file.read_text(encoding="utf-8"), self.original_text)

    def test_overwrite_replaces_the_experiment_file(self):
        result = self.config.create_experiment(
            name="vac",
            model="models.build_function.build_tlp",
            dataset="CSL-Daily",
            overwrite=True,
            num_epoch=7,
        )
        self.assertTrue(result["replaced_existing"])
        doc = yaml.safe_load(self.experiment_file.read_text(encoding="utf-8"))
        self.assertEqual(doc["model"], "models.build_function.build_tlp")
        self.assertEqual(doc["dataset"], "CSL-Daily")
        self.assertEqual(doc["num_epoch"], 7)
        # 只替换目标文件
        self.assertTrue((self.layout.configs_dir / "tlp.yaml").is_file())

    def test_overwrite_on_a_fresh_name_reports_no_replacement(self):
        """overwrite=True 只是「允许覆盖」:文件原本不存在,就没有替换任何东西。"""
        result = self.config.create_experiment(
            name="fresh",
            model="models.build_function.build_vac",
            dataset="phoenix2014",
            overwrite=True,
        )
        self.assertFalse(result["replaced_existing"])
        self.assertTrue((self.layout.configs_dir / "fresh.yaml").is_file())

    def test_illegal_experiment_name_is_rejected(self):
        for name in ("_private", "1abc", "with space"):
            with self.subTest(name=name):
                with self.assertRaises(McpToolError):
                    self.config.create_experiment(
                        name=name,
                        model="models.build_function.build_vac",
                        dataset="phoenix2014",
                    )


class MaterializeTests(unittest.TestCase):
    """临时运行配置:实验配置 + 覆盖 = 一份新文件,原文件不动。"""

    def setUp(self):
        self.layout = make_temp_repo()
        self.config = ExperimentConfig(self.layout)
        self.experiment_file = self.layout.configs_dir / "vac.yaml"
        self.original_text = self.experiment_file.read_text(encoding="utf-8")
        self.declared = yaml.safe_load(self.original_text)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_temp_config_sits_next_to_the_configs_dir(self):
        """必须与实验配置同目录:map() 按进程 cwd 找 ./configs/<dataset>.yaml。"""
        result = self.config.materialize_run_config("vac", {"num_epoch": 3}, run_id="r-loc")
        path = Path(result["path"])
        self.assertEqual(path.parent, self.layout.configs_dir)
        self.assertTrue(path.name.startswith(EPHEMERAL_PREFIX))
        self.assertTrue(path.is_file())
        self.assertEqual(result["name"], "vac")
        self.assertEqual(result["model"], "models.build_function.build_vac")
        self.assertEqual(result["dataset"], "phoenix2014")
        # 临时配置不会被当成一个实验
        self.assertNotIn(path.stem, self.config.experiment_names())

    def test_original_config_is_never_touched(self):
        self.config.materialize_run_config(
            "vac", {"num_epoch": 3, "model_args": {"use_bn": 0}}, run_id="r-untouched"
        )
        self.assertEqual(
            self.experiment_file.read_text(encoding="utf-8"),
            self.original_text,
            "materialize 不应改动实验自己的配置文件",
        )

    def test_nested_override_deep_merges_so_siblings_survive(self):
        result = self.config.materialize_run_config(
            "vac", {"model_args": {"use_bn": 0}}, run_id="r-nested"
        )
        doc = yaml.safe_load(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["model_args"]["use_bn"], 0)
        for key in ("num_classes", "c2d_type", "kernel_size", "stride"):
            self.assertEqual(
                doc["model_args"][key], self.declared["model_args"][key], f"{key} 被丢掉了"
            )
        # 没被覆盖的顶层键也原样保留
        self.assertEqual(doc["optimizer_args"], self.declared["optimizer_args"])
        self.assertEqual(doc["model"], self.declared["model"])

    def test_temp_config_passes_the_real_config_chain(self):
        result = self.config.materialize_run_config(
            "vac",
            {"num_epoch": 3, "optimizer_args": {"base_lr": 0.0002}},
            run_id="r-resolve",
        )
        verdict = self.config.resolve("vac", config_path=result["path"])
        self.assertTrue(verdict["ok"], verdict)
        effective = verdict["effective"]
        self.assertEqual(effective["num_epoch"], 3)
        self.assertEqual(effective["optimizer_args"]["base_lr"], 0.0002)
        # 只改 base_lr,optimizer_args 的兄弟键必须还在
        for key in ("optimizer", "weight_decay", "step", "nesterov"):
            self.assertEqual(
                effective["optimizer_args"][key], self.declared["optimizer_args"][key]
            )
        self.assertEqual(effective["model"], "models.build_function.build_vac")

    def test_snapshot_is_kept_for_reproducibility(self):
        result = self.config.materialize_run_config("vac", {"num_epoch": 3}, run_id="r-snap")
        snapshot = Path(result["snapshot"])
        self.assertTrue(snapshot.is_file())
        self.assertEqual(snapshot.parent, self.layout.runs_dir)
        self.assertEqual(snapshot.name, "r-snap.config.yaml")
        self.assertEqual(
            snapshot.read_text(encoding="utf-8"),
            Path(result["path"]).read_text(encoding="utf-8"),
        )

    def test_unknown_override_key_and_unknown_experiment_are_rejected(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.materialize_run_config("vac", {"no_such_key": 1}, run_id="r-bad")
        self.assertIn("no_such_key", str(ctx.exception))
        with self.assertRaises(McpToolError):
            self.config.materialize_run_config("not_an_experiment", {}, run_id="r-bad")
        self.assertFalse(self.config.ephemeral_config_path("r-bad").exists())

    def test_non_mapping_overrides_and_illegal_run_id_are_rejected(self):
        with self.assertRaises(McpToolError):
            self.config.materialize_run_config("vac", ["num_epoch"], run_id="r-bad")
        with self.assertRaises(McpToolError):
            self.config.materialize_run_config("vac", {"num_epoch": 1}, run_id="../escape")
        with self.assertRaises(McpToolError):
            self.config.materialize_run_config("vac", {"num_epoch": 1}, run_id="")

    def test_cleanup_only_removes_its_own_temp_files(self):
        kept = self.config.materialize_run_config("vac", {"num_epoch": 1}, run_id="keep")
        dropped = self.config.materialize_run_config("vac", {"num_epoch": 2}, run_id="drop")

        removed = self.config.sweep_ephemeral_configs(active_run_ids=["keep"])
        self.assertEqual(removed, [Path(dropped["path"]).name])
        self.assertTrue(Path(kept["path"]).is_file())

        # 实验自己的配置文件永远不属于清理范围
        self.assertFalse(self.config.cleanup_ephemeral_config(self.experiment_file))
        self.assertTrue(self.experiment_file.is_file())
        # 不在 configs/ 下的同名文件也不动
        outside = self.layout.runs_dir / f"{EPHEMERAL_PREFIX}outside.yaml"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("num_epoch: 1\n", encoding="utf-8")
        self.assertFalse(self.config.cleanup_ephemeral_config(outside))
        self.assertTrue(outside.is_file())


class RepoDiscoveryTests(unittest.TestCase):
    """定位仓库:干净的进程里要能自己找到,错误的 OPENCSLR_ROOT 要立刻报错。"""

    def test_discover_finds_the_repo_without_a_root_override(self):
        with mock.patch.dict(os.environ, env_without_root_override(), clear=True):
            layout = RepoLayout.discover()
        self.assertEqual(layout.root, REPO_ROOT)
        self.assertEqual(layout.configs_dir, REPO_ROOT / "core" / "configs")

    def test_discover_rejects_a_root_that_is_not_a_repo(self):
        not_a_repo = REPO_ROOT / "mcp_server"
        with mock.patch.dict(os.environ, {"OPENCSLR_ROOT": str(not_a_repo)}, clear=True):
            with self.assertRaises(FileNotFoundError) as ctx:
                RepoLayout.discover()
        self.assertIn("core/main.py", str(ctx.exception))


class ProblemReportingTests(unittest.TestCase):
    """坏配置必须在 list_experiments 的 problems 里说出来,不能静默通过。"""

    def setUp(self):
        self.layout = make_temp_repo()
        self.config = ExperimentConfig(self.layout)
        configs = self.layout.configs_dir
        (configs / "bad_model.yaml").write_text(
            "model: models.build_function.build_nope\ndataset: phoenix2014\n", encoding="utf-8"
        )
        (configs / "bad_dataset.yaml").write_text(
            "model: models.build_function.build_vac\ndataset: no_such_dataset\n", encoding="utf-8"
        )
        (configs / "bad_key.yaml").write_text(
            "model: models.build_function.build_vac\ndataset: phoenix2014\nnot_a_key: 1\n",
            encoding="utf-8",
        )
        (configs / f"{EPHEMERAL_PREFIX}leftover.yaml").write_text(
            "model: models.build_function.build_vac\ndataset: phoenix2014\n", encoding="utf-8"
        )

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_problems_point_at_the_actual_defect(self):
        items = {item["name"]: item for item in self.config.list_experiments()}

        self.assertEqual(items["vac"]["problems"], [])
        self.assertEqual(items["baseline"]["problems"], [])
        self.assertEqual(items["tlp"]["problems"], [])

        self.assertTrue(
            any("build_nope" in problem for problem in items["bad_model"]["problems"]),
            items["bad_model"]["problems"],
        )
        self.assertTrue(
            any("no_such_dataset" in problem for problem in items["bad_dataset"]["problems"]),
            items["bad_dataset"]["problems"],
        )
        self.assertTrue(
            any("not_a_key" in problem for problem in items["bad_key"]["problems"]),
            items["bad_key"]["problems"],
        )

    def test_ephemeral_and_dataset_configs_are_not_experiments(self):
        names = {item["name"] for item in self.config.list_experiments()}
        self.assertNotIn(f"{EPHEMERAL_PREFIX}leftover", names)
        self.assertNotIn("phoenix2014", names)
        self.assertNotIn("CSL-Daily", names)


class EmptyConfigsTests(unittest.TestCase):
    """configs/ 里什么都没有时,查询要如实回答空集,创建要拒绝而不是瞎猜。"""

    def setUp(self):
        self.layout = make_temp_repo(with_configs=False)
        self.config = ExperimentConfig(self.layout)

    def tearDown(self):
        remove_tree(self.layout.root)

    def test_queries_report_empty_sets(self):
        self.assertEqual(self.config.experiment_names(), [])
        self.assertEqual(self.config.dataset_names(), [])
        self.assertEqual(self.config.list_experiments(), [])
        self.assertEqual(self.config.list_options()["datasets"], [])

    def test_create_rejects_a_dataset_that_has_no_config(self):
        with self.assertRaises(McpToolError) as ctx:
            self.config.create_experiment(
                name="x", model="models.build_function.build_vac", dataset="phoenix2014"
            )
        self.assertIn("phoenix2014", str(ctx.exception))
        self.assertFalse((self.layout.configs_dir / "x.yaml").exists())


if __name__ == "__main__":
    unittest.main()
