# CLAUDE.md

Guidance for agents working in this repository.

## 这个仓库是什么

**上游 [immc-lab/OpenCSLR](https://github.com/immc-lab/OpenCSLR) 的原样代码 + 一层 MCP 服务。**

- `core/` 是上游的**逐字节拷贝**(验证:`diff -r -x __pycache__ <上游克隆>/core core` 应无输出);
- `docs/`、`README.md`、`LICENSE` 同样取自上游;
- 本仓库**只额外增加** `mcp_server/`(MCP 服务)与 `.mcp.json`(客户端注册)。

**铁律:`core/` 一行都不要改。** 需要改训练/模型/配置逻辑时,去上游改,再把上游新版
拷回来。这样做是为了不再出现「本地结构与上游漂移」——历史上正是这种漂移导致了
SEN 时序卷积、SlowFast FUSE、state_dict key 对不上等一连串问题。MCP 层要迁就上游,
而不是反过来。

## 分支

| 分支 | 内容 |
| --- | --- |
| `feat/upstream-core-mcp` | 上游 core 原样 + MCP(当前开发分支) |
| `legacy/refactored-mcp` | 旧的重构版 core(分节 exp.yaml/network.yaml + `modules/` 分层 + 注册表 + 运行时控制面)与当时的 MCP,仅作历史保留 |
| `main` | 未改动;合并方向待确认 |

## 上游的约定(与旧重构版差异很大,别按旧的写)

- **一个实验 = `core/configs/` 下一个扁平 YAML**(`vac.yaml` / `tlp.yaml` / `baseline.yaml`),
  顶层直接是参数;`model: models.build_function.build_vac` 是**模型点号路径**。
  没有 `--exp`,没有 `network.yaml`。一个实验一个文件,没有分节、没有 YAML anchor。
- `core/configs/` **同时**放着数据集配置(`phoenix2014.yaml` 等,含 `dataset_root`/`dict_path`)。
  判别规则:有 `model:` 的是实验,有 `dataset_root`/`dict_path` 的是数据集。
- 训练命令:`python main.py --config configs/<name>.yaml --phase train`;**cwd 必须是 `core/`**,
  因为 `ArgumentManager.map()` 会去读 `./configs/<dataset>.yaml`(相对进程 cwd)。
  `--model` 不必传,配置里写谁就用谁。
- `ArgumentManager.map(config)` 用 `assert k in argparse_dests` 检查顶层键,然后
  `set_defaults(**config)` + 重新 `parse()`。**它是整块替换字典**,所以任何「只改嵌套
  字典里一个键」的用法都必须先深合并。
- `ConfigManager` 只做 `yaml.load` + 字典合并,**没有任何校验**(没有嵌套键白名单、
  没有类型规则、`base_lr` 为负也接受)。它顶层 `import toml` 但从未使用。
- 上游目录名有拼写错误:`core/pipline/`(不是 `pipeline`),照抄不要「修」。
- 训练产物:日志 `<work_dir>/log_<时间戳>.log`(loguru);checkpoint
  `<work_dir>_best_model.pt` 与 `<work_dir>dev_<wer>_epoch<n>_model.pt`(`work_dir` 同时
  被当目录和文件名前缀)。**上游不写任何结果 JSON**。
- 没有 `--control-file`,没有运行时控制面:训练中途改不了超参数。

## MCP 服务(`mcp_server/`)

设计原则:**MCP 是上游代码之上的适配层**,不复制上游的规则,也不要求上游为它改动。

- `server.py` 是**唯一**导入 `mcp` 的模块;其余(config / runs / results / paths / core_probe)
  都是纯 Python,没有 torch 也能测。
- `core_probe.py` 在子进程里跑**上游真实的** `ArgumentManager.init → ConfigManager.init →
  map()`,回传生效配置与参数 schema。绝不在这里复刻优先级或校验规则——上游改了,
  这里跟着变。探针只导入 `manager/argument_manager.py` 与 `manager/config_manager.py`
  (不需要 torch),并在缺 `toml` 时注入空桩(上游 import 了它却不用)。
- 工具通过 `server.tool()` 注册(不是 `mcp.tool()`):它把 `McpToolError` 翻译成 SDK 的
  `ToolError`,这是 SDK 唯一会原样透出 message 的异常类型;其它异常会被压成一句
  `Error executing tool <name>`,「实验不存在 / model 路径无效」这类可行动信息就丢了。
- `launch_experiment(overrides=...)` 把「实验配置 + 覆盖」落成
  `core/configs/_mcp_run_<run_id>.yaml` 再启动,**不改动实验自己的配置文件**;字典类
  覆盖按 YAML 语义深合并(见上:上游 `map` 是整块替换)。临时配置进程结束即清理,
  快照留在 `<runs_dir>/<run_id>.config.yaml`。
- 进程脱离会话启动(`start_new_session=True`),记录与日志在 `<repo>/.mcp_runs/`;
  `stop_run` 先核对 pid 的命令行再发信号,绝不误杀无关进程。
- **能力边界要如实**:上游没有的能力就不要造工具。历史版本里的
  `set_hyperparameters` / `get_control_state`(依赖给上游打 `--control-file` 补丁)在
  本分支已删除;`get_hyperparameters` 的 `hot` 与嵌套 `allowed` 如实为 `null`;
  上游不落盘结果 JSON,`get_results` 就从 `log_*.log` 解析 WER 并用 `source` 标明来源。
- 共 16 个工具(清单见 `mcp_server/README.md`)。

## 常用命令

```bash
# MCP 服务
python3 -m mcp_server --root "$PWD"                      # stdio
python3 -m unittest discover -s mcp_server/tests -t .     # 全量测试(无需 torch/GPU)

# 训练(上游原样)
cd core && python main.py --config configs/vac.yaml --phase train
```

## MCP 相关环境变量

| 变量 | 作用 |
| --- | --- |
| `OPENCSLR_ROOT` | 仓库根目录(默认向上查找 `core/main.py`) |
| `OPENCSLR_MCP_RUNS_DIR` | 运行记录目录(默认 `<repo>/.mcp_runs`) |
| `OPENCSLR_PYTHON` | 执行 `core/main.py` 的解释器;服务本身不需要 torch,起训练时才用它 |
