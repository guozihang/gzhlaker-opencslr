# OpenCSLR MCP 服务

把仓库的实验管理能力封装成 [MCP](https://modelcontextprotocol.io) 工具,让
**已有的智能体**(Claude Code、IDE 助手、任何 MCP 客户端)通过调用工具就能管理
本仓库的实验:看清有哪些实验、改配置、起训练、追进度、读 WER 与 checkpoint。

这里的定位是「把仓库 MCP 化」——仓库本身不内置任何智能体、不调用任何大模型
API;决策由客户端那侧的智能体做,本服务只负责把仓库能力可靠地暴露出去。

> **本服务的适配对象是上游 [immc-lab/OpenCSLR](https://github.com/immc-lab/OpenCSLR)
> 原样的 `core/`**(本分支的 `core/` 是它的逐字节拷贝,**没有任何补丁**)。
> 因此有些能力取决于上游有没有,不取决于我们想不想要——见下面「能力边界」。

## 快速开始

```bash
# 1. 安装依赖(只多一个 mcp 包,不影响训练环境)
pip install -r mcp_server/requirements.txt

# 2. 注册到 MCP 客户端(以 Claude Code 为例,在仓库根目录执行)
claude mcp add opencslr -- python3 -m mcp_server --root "$PWD"

# 3. 验证
claude mcp list
```

仓库根目录已提供 `.mcp.json`(项目级配置),在仓库里打开 Claude Code 时会自动
发现名为 `opencslr` 的服务。

手动启动(调试用):

```bash
python3 -m mcp_server                      # stdio,协议走 stdout,日志走 stderr
python3 -m mcp_server --transport sse --port 8765 --host 0.0.0.0   # 局域网共用
```

环境变量:

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `OPENCSLR_ROOT` | 仓库根目录 | 从当前目录向上查找 `core/main.py` |
| `OPENCSLR_MCP_RUNS_DIR` | 运行记录目录 | `<repo>/.mcp_runs` |
| `OPENCSLR_PYTHON` | 执行 `core/main.py` 的解释器 | 本服务进程的解释器 |

> 探针(`resolve_experiment` / `get_hyperparameters`)会导入上游的
> `manager/argument_manager.py` 与 `manager/config_manager.py`,它们需要 PyYAML;
> 上游 `config_manager` 顶层还 `import toml` 但**从不使用**,所以探针在缺 toml 时
> 会注入一个空桩,不必为了探针去装它。真正起训练仍然需要上游完整的依赖
> (torch / loguru / rich / wandb / toml 等),用 `OPENCSLR_PYTHON` 指向训练环境即可。

## 上游的配置模型(与本服务的对应关系)

- **一个实验 = `core/configs/` 下一个扁平 YAML**(`vac.yaml`、`tlp.yaml`、`baseline.yaml`),
  顶层直接是参数,`model: models.build_function.build_vac` 是**模型点号路径**;
- `core/configs/` 里**同时**放着数据集配置(`phoenix2014.yaml` 等,含
  `dataset_root`/`dict_path`)。本服务用 `model:` 键区分:有它的是实验,没有的是数据集;
- 训练命令是 `python main.py --config configs/<name>.yaml --phase train`(**没有 `--exp`**);
  `--model` 不必传,配置里写谁就用谁;
- 上游 `ArgumentManager.map()` 会去读 `./configs/<dataset>.yaml`(相对**进程 cwd**),
  所以本服务一律以 `core/` 为 cwd 启动子进程——这是硬要求,不是习惯。

## 工具一览

### 看清现状

| 工具 | 说明 |
| --- | --- |
| `get_server_info` | 服务连的是哪个仓库、用哪个解释器、配置目录与运行记录目录在哪 |
| `list_experiments` | `configs/` 下的实验清单(模型点号路径、数据集、work_dir、静态问题) |
| `get_experiment_config` | 某个实验**声明**了什么(配置文件原文 + 静态问题清单) |
| `list_options` | 可选的实验名、模型点号路径、数据集名 |
| `resolve_experiment` | 某个实验**实际生效**的完整配置(跑上游真实初始化链) |
| `get_hyperparameters` | 参数表 + 当前生效值 + 类型/默认值(取值域与热改能力见「能力边界」) |

### 跑实验

| 工具 | 说明 |
| --- | --- |
| `create_experiment` | 在 `configs/` 下新建实验配置文件,写入前静态校验 model/dataset/键名 |
| `launch_experiment` | 启动训练/评估,立刻返回 `run_id`;`overrides` 可带超参数覆盖 |
| `launch_preprocess` | 启动数据预处理(可用 `script` 指定上游按数据集分的脚本变体) |
| `stop_run` | 停止自己启动的进程(SIGTERM,可选 SIGKILL) |

### 看结果

| 工具 | 说明 |
| --- | --- |
| `list_runs` | 本服务启动过的运行及当前状态 |
| `get_run_status` | 单个运行的进程状态 + 日志末尾 + 结果 + checkpoint |
| `get_results` | work_dir 的 WER(上游不落盘 JSON,从 `log_*.log` 解析,`source` 字段说明来源) |
| `list_artifacts` | checkpoint、结果文件、日志 |
| `tail_log` | 日志末尾若干行(按 run_id 或 work_dir) |
| `list_work_dirs` | 找出跑过的实验目录 |

共 16 个工具。

## 能力边界(上游没有的,这里如实没有)

`core/` 逐字用上游、不加补丁,所以下面这些**做不到**,工具会如实说明而不是假装:

| 能力 | 上游现状 | 本服务的表现 |
| --- | --- | --- |
| **训练中途热改超参数** | 没有运行时控制面(无 `--control-file`) | **没有** `set_hyperparameters` / `get_control_state` 工具;`get_hyperparameters` 的 `hot` 一律为 `null` 并给出原因 |
| **结果 JSON** | `ExperimentManager` 不写任何结果 JSON | `get_results` 从 `work_dir/log_*.log` 解析 `{'Dev': ..}` / `Best_dev: ..` 等行;`source` 标为 `log`;日志里才是权威值 |
| **嵌套键/取值校验** | `ConfigManager` 只 `yaml.load` + 合并,连 `base_lr` 为负也照收 | `get_hyperparameters` 的嵌套节只给当前值,`allowed` 为 `null`;**不假装有校验** |
| **模型名解析** | `model:` 是点号路径,`importlib` 动态导入 | `create_experiment` / `list_experiments` 做**静态**定位(文件 + 属性名)拦住拼写错误;真正的导入失败只在启动时暴露 |

## 设计要点

**配置只有一个真相来源。** 优先级与「键必须是 argparse 参数」的检查都只存在于上游
`manager/argument_manager.py`;本服务不抄第二份,而是通过 `core_probe.py` 以子进程
方式跑上游真实的 `ArgumentManager` + `ConfigManager`,因此 `resolve_experiment`
的结论与真正启动时一致。

**覆盖项要深合并。** `launch_experiment(overrides=...)` 不会改动实验自己的配置文件,
而是把「实验配置 + 覆盖」落成 `configs/_mcp_run_<run_id>.yaml` 再启动它。上游
`map()` 是 `set_defaults(**config)`,字典整块替换会丢掉原文件里的兄弟键
(只想改 `model_args.use_bn` 却丢了 `num_classes`),所以字典类覆盖在这里按 YAML
语义深合并。临时配置在进程结束后自动清理,快照留在 `<runs_dir>/<run_id>.config.yaml`。

**报错要报得出原因。** 工具抛 `McpToolError` 时会先翻译成 SDK 自己的 `ToolError`
(`server.tool()` 这层包装做的就是这个)。不翻译的话 SDK 只把消息压成一句
`Error executing tool <name>`,「实验不存在 / model 路径无效 / 配置解析失败」这类
可据以行动的信息就丢了。

**长训练不阻塞。** `launch_experiment` 起进程后立刻返回 `run_id`,进程输出重定向到
`<runs_dir>/<run_id>.log`,命令、pid、work_dir、用的是哪份配置记在同名 JSON 里。

**只碰自己启动的进程。** `stop_run` 先核对 pid 对应的命令行里确实有 `main.py`/
`dataset_preprocess.py` 和实验名,对不上就拒绝发信号。

**产物命名两种形态都认。** `work_dir` 在上游 `ExperimentManager` 里同时被当作目录
(放日志)和文件名前缀(放 checkpoint),所以 checkpoint 可能在 `<work_dir>_best_model.pt`,
也可能是同级的 `<work_dir>dev_19.20_epoch5_model.pt`,`list_artifacts` 都会列出来。

## 目录结构

```
mcp_server/
├── __main__.py       # 入口:python -m mcp_server [--transport ...]
├── server.py         # MCP 工具定义(唯一依赖 mcp 包的模块)
├── config.py         # configs/ 的读取与写入 + 临时运行配置 + 静态校验
├── core_probe.py     # 子进程里跑上游真实配置链,回传生效配置与参数 schema
├── runs.py           # 启动/停止/状态机 + 运行记录
├── results.py        # 日志解析(WER)、checkpoint、目录发现
├── paths.py          # 仓库目录布局
└── tests/            # 不需要 torch / GPU 的单元与集成测试
```

## 测试

```bash
python -m unittest discover -s mcp_server/tests -t .
```

用例只依赖标准库与 PyYAML:用假的 `main.py` 覆盖进程生命周期(启动、日志重定向、
正常结束、异常退出、主动停止、拒绝误杀),配置解析用的是**本仓库 `core/` 里上游
原样的** `manager/argument_manager.py` 与 `manager/config_manager.py`。

其中 `test_mcp_protocol.py` 是端到端的一层:真的用 MCP 客户端连上 `python -m mcp_server`,
验证工具注册、参数 schema、正常返回,以及出错时原因确实送到了调用方(未装 `mcp` 包时
整类跳过)。

## 分支关系

本仓库有三个分支:

| 分支 | 内容 |
| --- | --- |
| `legacy/refactored-mcp` | 重构版 core(分节 exp.yaml/network.yaml + `modules/` 分层 + 注册表 + 运行时控制面)与当时的 MCP |
| `feat/upstream-core-mcp` | **本分支**:上游 `core/` 逐字节原样 + 按上游约定重写的 MCP(无热改、结果读日志) |
| `main` | 保持原样,待确认后再决定合并方向 |

之所以不再「把上游代码改造成本地结构」,是因为那会持续产生结构性偏离;反过来,
把 MCP 建成**上游代码之上的适配层**,上游怎么改都只需要跟着改适配层。
