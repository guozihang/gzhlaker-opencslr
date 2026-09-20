# OpenCSLR MCP 服务

把仓库的实验管理能力封装成 [MCP](https://modelcontextprotocol.io) 工具,让
**已有的智能体**(Claude Code、IDE 助手、任何 MCP 客户端)通过调用工具就能管理
本仓库的实验:看清有哪些实验、改配置、起训练、追进度、读 WER 与 checkpoint。

这里的定位是「把仓库 MCP 化」——仓库本身不内置任何智能体、不调用任何大模型
API;决策由客户端那侧的智能体做,本服务只负责把仓库能力可靠地暴露出去。

## 快速开始

```bash
# 1. 安装依赖(只多一个 mcp 包,不影响训练环境)
pip install -r mcp_server/requirements.txt

# 2. 注册到 MCP 客户端(以 Claude Code 为例,在仓库根目录执行)
claude mcp add opencslr -- python3 "$PWD/mcp_server/server.py"
#   或使用 stdio 模块入口:
claude mcp add opencslr -- python3 -m mcp_server --root "$PWD"

# 3. 验证
claude mcp list
```

仓库根目录已提供 `.mcp.json`(项目级配置),在仓库里打开 Claude Code 时会自动
发现名为 `opencslr` 的服务;也可以把上面的 `claude mcp add` 换成自己的客户端写法。

手动启动(调试用):

```bash
python3 -m mcp_server                      # stdio,协议走 stdout,日志走 stderr
python3 -m mcp_server --transport sse --port 8765 --host 0.0.0.0   # 局域网共用
```

环境变量:

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `OPENCSLR_ROOT` | 仓库根目录 | 从当前目录向上查找 `core/configs/exp.yaml` |
| `OPENCSLR_MCP_RUNS_DIR` | 运行记录目录 | `<repo>/.mcp_runs` |
| `OPENCSLR_PYTHON` | 执行 `core/main.py` 的解释器 | 本服务进程的解释器 |

> 如果 MCP 服务跑在没装 torch 的解释器下,用 `OPENCSLR_PYTHON` 指向训练环境的
> python 即可——**服务本身不需要 torch**,只有真正起训练时才用得到它。

## 工具一览

### 看清现状

| 工具 | 说明 |
| --- | --- |
| `get_server_info` | 服务连的是哪个仓库、用哪个解释器、运行记录目录在哪 |
| `list_experiments` | exp.yaml 里的实验清单(网络/数据集/work_dir/配置问题) |
| `get_experiment_config` | 某个实验**声明**了什么(exp 节 + 网络节 + 数据集节) |
| `list_options` | 可选的实验名、网络名、数据集名、模型名 |
| `resolve_experiment` | 某个实验**实际生效**的完整配置,并做启动前校验 |

### 跑实验

| 工具 | 说明 |
| --- | --- |
| `create_experiment` | 在 exp.yaml 里新建(或整节替换)实验节 |
| `launch_experiment` | 启动训练/评估,立刻返回 `run_id` |
| `launch_preprocess` | 启动数据集预处理 |
| `stop_run` | 停止自己启动的进程(SIGTERM,可选 SIGKILL) |

### 看结果

| 工具 | 说明 |
| --- | --- |
| `list_runs` | 本服务启动过的运行及当前状态 |
| `get_run_status` | 单个运行的进程状态 + 日志末尾 + 结果 + checkpoint |
| `get_results` | work_dir 下的 WER / 样本统计 JSON |
| `list_artifacts` | checkpoint、结果文件、日志 |
| `tail_log` | 日志末尾若干行(按 run_id 或 work_dir) |
| `list_work_dirs` | 找出跑过的实验目录 |

## 设计要点

**配置只有一个真相来源。** 命令行 > YAML > 代码默认值 的优先级、未知键检查、嵌套
键与取值校验,规则都只写在 `core/manager` 里。MCP 层不抄第二份,而是通过
`core_probe.py` 以子进程方式跑真实的 `ArgumentManager` + `ConfigManager`,因此
`resolve_experiment` 的结论与真正启动时的行为一致——配置错了在占上 GPU 之前就报错。

**报错要报得出原因。** 工具抛 `McpToolError` 时会先翻译成 SDK 自己的 `ToolError`
(`server.tool()` 这层包装做的就是这个)。不翻译的话 SDK 只把消息压成一句
`Error executing tool <name>`,「实验不存在 / network 未定义 / 配置校验失败」这类
可据以行动的信息就丢了——对调用方来说,等于没有报错。

**长训练不阻塞。** `launch_experiment` 起进程后立刻返回 `run_id`,进程输出重定向到
`<runs_dir>/<run_id>.log`,命令、pid、work_dir 记在同名 JSON 里。服务重启后仍能
按 pid 判断死活(只是拿不到退出码时会标成 `unknown`)。

**只碰自己启动的进程。** `stop_run` 先核对 pid 对应的命令行里确实有 `main.py`/
`dataset_preprocess.py` 和实验名,对不上就拒绝发信号。

**产物命名两种形态都认。** `work_dir` 在 `ExperimentManager` 里同时被当作目录
(放日志与结果)和文件名前缀(放 checkpoint),所以 checkpoint 可能在
`<work_dir>/_best_model.pt`,也可能是同级的 `<work_dir>_best_model.pt`,`list_artifacts`
两种都会列出来。

## 目录结构

```
mcp_server/
├── __main__.py       # 入口:python -m mcp_server [--transport ...]
├── server.py         # MCP 工具定义(唯一依赖 mcp 包的模块)
├── config.py         # 三个配置入口的读取与写入
├── core_probe.py     # 子进程里跑真实配置管理器,回传生效配置
├── runs.py           # 启动/停止/状态机 + 运行记录
├── results.py        # 结果、日志、checkpoint 读取
├── paths.py          # 仓库目录布局
└── tests/            # 不需要 torch / GPU 的单元与集成测试
```

## 测试

```bash
python -m unittest discover -s mcp_server/tests -t .
```

用例只依赖标准库与 PyYAML:用假的 `main.py` 覆盖进程生命周期(启动、日志重定向、
正常结束、异常退出、主动停止、拒绝误杀),配置解析用的是真实 `core/manager` 代码。

其中 `test_mcp_protocol.py` 是端到端的一层:真的用 MCP 客户端连上 `python -m mcp_server`,
验证工具注册、参数 schema、正常返回,以及出错时原因确实送到了调用方(未装 `mcp` 包时
整类跳过)。

## 与旧 `agent/` 目录的关系

此前的 `agent/` 目录走的是「把仓库包装成一个智能体」:自己实现 DeepSeek 客户端、
动作校验器、预算守卫、记忆库,由仓库内部去驱动大模型。那条路已被放弃
(相关代码已删除),原因是它要求仓库内置一个智能体运行时,能力与模型耦合,
也无法被已有智能体复用。

现在改为 MCP 化:MCP 服务只做「仓库能力的标准化出口」,不调用任何模型,谁来做决策
由客户端决定。原来 `agent/` 里有价值的部分以另一种形式保留了下来:

- 动作校验 → `resolve_experiment`(复用 core 的真实校验,而不是另写一份 schema)
- 预算/资源守卫 → 由调用方自行判断,服务如实报告 work_dir、日志与进程状态
- 实验记忆/去重 → `list_runs`、`list_work_dirs`、`get_results` 提供事实依据
