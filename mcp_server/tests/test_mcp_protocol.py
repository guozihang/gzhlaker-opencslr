# -*- encoding: utf-8 -*-
"""端到端:真的用 MCP 客户端连上服务,确认工具能用。

这一层验证的是「已有智能体通过 MCP 管理实验」这条链路本身——协议握手、工具
注册、参数与返回值的序列化、错误上报。没装 mcp 包时整类跳过。
"""

import asyncio
import json
import sys
import unittest

from .helpers import REPO_ROOT

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    HAVE_MCP = True
except ImportError:  # 没装 MCP SDK 时跳过
    HAVE_MCP = False

EXPECTED_TOOLS = {
    "get_server_info",
    "list_experiments",
    "get_experiment_config",
    "list_options",
    "resolve_experiment",
    "create_experiment",
    "launch_experiment",
    "launch_preprocess",
    "stop_run",
    "list_runs",
    "get_run_status",
    "get_results",
    "list_artifacts",
    "tail_log",
    "list_work_dirs",
}

READ_ONLY_CALLS = [
    ("get_server_info", {}),
    ("list_experiments", {}),
    ("list_options", {}),
    ("get_experiment_config", {"name": "vac"}),
    ("resolve_experiment", {"name": "vac"}),
    ("resolve_experiment", {"name": "vac", "overrides": {"num_epoch": 3}}),
    ("list_runs", {}),
    ("list_work_dirs", {}),
    ("get_results", {"work_dir": "./work_dir/never_ran/"}),
    ("launch_experiment", {"name": "vac", "dry_run": True}),
]

# name, arguments, 期望错误信息里出现的关键词
ERROR_CALLS = [
    ("get_experiment_config", {"name": "does_not_exist"}, "不存在"),
    ("tail_log", {"run_id": "20260101-000000-nonexistent"}, "找不到运行记录"),
    ("create_experiment", {"name": "x", "network": "no_such_net", "dataset": "phoenix2014"}, "network"),
    ("launch_experiment", {"name": "no_such_exp"}, "不存在"),
]


@unittest.skipUnless(HAVE_MCP, "需要安装 mcp 包(pip install -r mcp_server/requirements.txt)")
class ProtocolTests(unittest.TestCase):
    """一次会话里跑完所有只读调用,避免反复起进程。"""

    @classmethod
    def setUpClass(cls):
        cls.timeout = 120
        cls.session_result = asyncio.run(asyncio.wait_for(_exercise_server(), timeout=cls.timeout))

    def test_expected_tools_are_registered(self):
        schemas, _ = self.session_result
        missing = EXPECTED_TOOLS - set(schemas)
        self.assertEqual(missing, set(), f"缺少工具: {sorted(missing)}")

    def test_read_only_tools_return_json(self):
        _, results = self.session_result
        for name, payload in results.items():
            if name.startswith("error: "):
                continue  # 错误路径另有用例
            with self.subTest(tool=name):
                self.assertFalse(payload["is_error"], payload["text"][:300])
                self.assertIsInstance(json.loads(payload["text"]), dict)

    def test_resolve_reports_real_values(self):
        _, results = self.session_result
        resolved = json.loads(results["resolve_experiment"]["text"])
        self.assertTrue(resolved["ok"], resolved)
        self.assertEqual(resolved["key_settings"]["model"], "vac")
        self.assertEqual(resolved["effective"]["num_epoch"], 80)

    def test_override_is_visible_in_resolution(self):
        _, results = self.session_result
        resolved = json.loads(results["resolve_experiment (overridden)"]["text"])
        self.assertEqual(resolved["effective"]["num_epoch"], 3)

    def test_error_message_reaches_the_caller(self):
        """出错时调用方要看到原因,而不只是一句 "Error executing tool <name>"。

        这正是把 McpToolError 翻译成 SDK ToolError 的意义所在:普通异常会被
        SDK 压成不含原因的通用信息,调用方就只能瞎猜。
        """
        _, results = self.session_result
        checked = 0
        for label, payload in results.items():
            if not label.startswith("error: "):
                continue
            checked += 1
            with self.subTest(call=label):
                self.assertTrue(payload["is_error"])
                self.assertIn(payload["keyword"], payload["text"])
                generic = f"Error executing tool {label.split()[1]}"
                self.assertGreater(
                    len(payload["text"]), len(generic) + 2, payload["text"]
                )
        self.assertEqual(checked, len(ERROR_CALLS))

    def test_tool_schemas_keep_their_parameters(self):
        """注册包装不能把函数签名弄丢,否则调用方不知道要传什么。"""
        schemas, _ = self.session_result
        created = schemas["create_experiment"]["properties"]
        for key in ("name", "network", "dataset", "overrides", "work_dir", "overwrite"):
            self.assertIn(key, created)
        self.assertIn("required", schemas["create_experiment"])
        self.assertEqual(schemas["create_experiment"]["required"], ["name", "network", "dataset"])


async def _exercise_server():
    """连上 stdio 服务,列工具并真的调用它们。"""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server", "--root", str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
    )
    results = {}
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            schemas = {tool.name: tool.input_schema for tool in tools.tools}
            for name, arguments in READ_ONLY_CALLS:
                label = f"{name} (overridden)" if arguments.get("overrides") else name
                results[label] = _capture(await session.call_tool(name, arguments))
            for name, arguments, keyword in ERROR_CALLS:
                payload = _capture(await session.call_tool(name, arguments))
                payload["keyword"] = keyword
                results[f"error: {name} {arguments}"] = payload
    return schemas, results


def _capture(response):
    return {"is_error": bool(response.is_error), "text": _first_text(response)}


def _first_text(response):
    for item in response.content or []:
        text = getattr(item, "text", None)
        if text:
            return text
    return ""


if __name__ == "__main__":
    unittest.main()
