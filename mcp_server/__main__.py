# -*- encoding: utf-8 -*-
"""MCP 服务入口。

用法(在仓库根目录):
    python -m mcp_server                      # stdio,给本机 MCP 客户端用
    python -m mcp_server --transport sse --port 8765   # 局域网内多个客户端共用

stdio 模式下协议走 stdout,所有提示都写到 stderr。
"""

import argparse
import sys

from .errors import McpToolError

try:
    from .server import configure, mcp
except ImportError as import_error:  # 最常见的原因是没装 MCP SDK
    print(
        "[mcp_server] 无法导入 MCP SDK({});请先安装依赖:\n"
        "    pip install -r mcp_server/requirements.txt".format(import_error),
        file=sys.stderr,
    )
    raise SystemExit(2)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mcp_server",
        description="OpenCSLR 实验管理 MCP 服务",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="OpenCSLR 仓库根目录;默认从当前目录向上查找 core/configs/exp.yaml",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="传输方式;本机客户端用默认的 stdio",
    )
    parser.add_argument("--host", default=None, help="sse/http 监听地址(默认 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="sse/http 监听端口(默认 8000)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        layout = configure(args.root)
    except (FileNotFoundError, McpToolError) as exc:
        print(f"[mcp_server] 初始化失败: {exc}", file=sys.stderr)
        return 2

    print(
        f"[mcp_server] repo={layout.root} | core={layout.core_dir} | "
        f"python={layout.python} | transport={args.transport}",
        file=sys.stderr,
    )
    mcp.run(transport=args.transport, **listen_kwargs(args))
    return 0


def listen_kwargs(args):
    """把 --host/--port 变成对应 SDK 版本的传参方式。

    SDK 1.x 是 ``mcp.settings.host/port``,2.x 起改为 ``run(..., host=, port=)``。
    """
    if not args.host and not args.port:
        return {}
    settings = getattr(mcp, "settings", None)
    if settings is None:
        kwargs = {}
        if args.host:
            kwargs["host"] = args.host
        if args.port:
            kwargs["port"] = args.port
        return kwargs
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    return {}


if __name__ == "__main__":
    sys.exit(main())
