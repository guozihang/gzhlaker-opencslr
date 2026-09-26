# -*- encoding: utf-8 -*-
"""OpenCSLR MCP 服务。

把仓库的实验管理能力(MCP 工具)暴露给外部智能体,使已有的 MCP 客户端
(Claude Code、IDE 助手等)无需了解仓库内部结构即可管理实验。

分层:
    paths.py       仓库目录布局(纯路径计算)
    config.py      实验配置的读取与创建
    runs.py        运行记录、启动、停止、状态
    results.py     结果文件、日志、checkpoint 的读取
    core_probe.py  在 core/ 环境里复用真实配置管理器做解析与校验
    server.py      MCP 工具注册(唯一依赖 mcp 包的模块)

除 server.py 外,其余模块只依赖标准库与 PyYAML,不导入 core/ 下的任何
模块(因此不依赖 torch),以便在没有 GPU 的机器上直接跑测试。
"""

__all__ = ["paths", "config", "runs", "results", "server"]
