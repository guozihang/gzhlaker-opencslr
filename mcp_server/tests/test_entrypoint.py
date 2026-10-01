# -*- encoding: utf-8 -*-
"""入口层:--host/--port 的 SDK 版本适配,以及 `python mcp_server/server.py` 这条用法。

这两件事都真的坏过:
  - SDK 2.x 里 settings 仍然存在(只剩 log_level 等字段),host/port 已改由
    ``run(..., host=, port=)`` 传入。按「有没有 settings」判版本会去写不存在的
    字段,服务启动即 ``ValueError: "Settings" object has no field "host"``;
  - README 把 ``python mcp_server/server.py`` 列为注册方式之一,而相对导入让它在
    直接执行时 ``ImportError: attempted relative import with no known parent package``。
"""

import asyncio
import inspect
import sys
import unittest
from types import SimpleNamespace

from .helpers import REPO_ROOT

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    from mcp_server import __main__ as entry

    HAVE_MCP = True
except ImportError:  # 没装 MCP SDK 时跳过
    HAVE_MCP = False


@unittest.skipUnless(HAVE_MCP, "需要安装 mcp 包(pip install -r mcp_server/requirements.txt)")
class ListenKwargsTests(unittest.TestCase):
    """--host/--port 必须送到当前 SDK 真正会读的地方。"""

    def setUp(self):
        # listen_kwargs 读的是 __main__ 里的模块级 mcp,用例自己换掉它
        self.original = entry.mcp
        self.addCleanup(setattr, entry, "mcp", self.original)

    def test_sdk2_settings_without_host_becomes_run_kwargs(self):
        entry.mcp = SimpleNamespace(settings=SimpleNamespace(log_level="INFO"))
        kwargs = entry.listen_kwargs(SimpleNamespace(host="0.0.0.0", port=8765))
        self.assertEqual(kwargs, {"host": "0.0.0.0", "port": 8765})

    def test_sdk2_never_writes_settings_host(self):
        """回归:曾经在这里写 settings.host,启动直接抛 ValueError。"""
        settings = SimpleNamespace(log_level="INFO")
        entry.mcp = SimpleNamespace(settings=settings)
        entry.listen_kwargs(SimpleNamespace(host="127.0.0.1", port=8765))
        self.assertFalse(hasattr(settings, "host"))

    def test_sdk1_settings_with_host_is_updated_in_place(self):
        settings = SimpleNamespace(host="127.0.0.1", port=8000)
        entry.mcp = SimpleNamespace(settings=settings)
        kwargs = entry.listen_kwargs(SimpleNamespace(host="0.0.0.0", port=9000))
        self.assertEqual(kwargs, {}, "SDK 1.x 走 settings,不该再往 run() 传参")
        self.assertEqual((settings.host, settings.port), ("0.0.0.0", 9000))

    def test_sdk1_only_touches_the_flags_that_were_given(self):
        settings = SimpleNamespace(host="127.0.0.1", port=8000)
        entry.mcp = SimpleNamespace(settings=settings)
        entry.listen_kwargs(SimpleNamespace(host="0.0.0.0", port=None))
        self.assertEqual((settings.host, settings.port), ("0.0.0.0", 8000))

    def test_no_endpoint_flags_return_nothing(self):
        entry.mcp = SimpleNamespace(settings=SimpleNamespace(log_level="INFO"))
        self.assertEqual(entry.listen_kwargs(SimpleNamespace(host=None, port=None)), {})

    def test_real_sdk_accepts_whatever_we_return(self):
        """真正装着的 SDK 上,返回的 kwargs 必须是它认得的参数。"""
        from mcp_server.server import mcp as real_mcp

        entry.mcp = real_mcp
        kwargs = entry.listen_kwargs(SimpleNamespace(host="127.0.0.1", port=8765))
        runner = getattr(real_mcp, "run_sse_async", None)
        accepted = set(inspect.signature(runner).parameters) if runner else set()
        for key in kwargs:
            self.assertIn(key, accepted, f"{key} 不是 run_sse_async 能接受的参数")


@unittest.skipUnless(HAVE_MCP, "需要安装 mcp 包(pip install -r mcp_server/requirements.txt)")
class ScriptEntryPointTests(unittest.TestCase):
    """README 给出的 `python mcp_server/server.py` 必须真能起来并注册工具。"""

    def test_server_script_speaks_mcp_over_stdio(self):
        from .test_mcp_protocol import EXPECTED_TOOLS

        script = REPO_ROOT / "mcp_server" / "server.py"
        version, names = asyncio.run(
            asyncio.wait_for(_handshake([str(script), "--root", str(REPO_ROOT)]), 60)
        )
        self.assertEqual(EXPECTED_TOOLS - set(names), set(), f"缺少工具: {sorted(EXPECTED_TOOLS - set(names))}")
        self.assertTrue(version, "server 版本号为空,MCP 客户端看不到服务标识")


async def _handshake(argv):
    """连一次 stdio 服务,返回 (版本号, 工具名列表)。"""
    parameters = StdioServerParameters(
        command=sys.executable, args=argv, cwd=str(REPO_ROOT)
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
    info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
    return getattr(info, "version", "") or "", [tool.name for tool in tools.tools]


if __name__ == "__main__":
    unittest.main()
