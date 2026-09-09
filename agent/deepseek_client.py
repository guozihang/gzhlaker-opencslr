# -*- encoding: utf-8 -*-
"""DeepSeek API 客户端。

使用 OpenAI 兼容的 API 接口调用 DeepSeek 模型。
支持 prompt caching 和结构化 JSON 输出。
"""

import os
import json
from typing import Dict, List, Optional, Any
from openai import OpenAI


class DeepSeekClient:
    """DeepSeek API 客户端。

    封装 DeepSeek Chat 和 Reasoner 模型的调用。
    """

    def __init__(self,
                 model: str = "deepseek-chat",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0):
        """初始化客户端。

        Args:
            model: 模型名称（'deepseek-chat' 或 'deepseek-reasoner'）
            api_key: API 密钥（默认从环境变量读取）
            temperature: 温度参数（0.0 = 完全确定性）
        """
        self.model = model
        self.temperature = temperature

        # 从环境变量获取 API key
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek API key not found. "
                    "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
                )

        # 初始化 OpenAI 客户端（指向 DeepSeek 端点）
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

        # 统计信息
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def chat(self,
             system_prompt: str,
             user_message: str,
             response_format: Optional[str] = None,
             max_tokens: int = 4096) -> Dict[str, Any]:
        """发送聊天请求。

        Args:
            system_prompt: 系统提示
            user_message: 用户消息
            response_format: 响应格式（'json_object' 或 None）
            max_tokens: 最大生成 token 数

        Returns:
            包含响应和统计信息的字典
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 构建请求参数
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens
        }

        # 添加结构化输出格式
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        # 调用 API
        response = self.client.chat.completions.create(**kwargs)

        # 提取响应
        content = response.choices[0].message.content

        # 如果是 JSON 格式，解析
        if response_format == "json_object":
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON response: {e}")
                content = {"error": "Invalid JSON", "raw": content}

        # 统计 token 使用
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # 计算成本（DeepSeek Chat 价格：$0.14/M input, $0.28/M output）
        cost = (input_tokens * 0.14 + output_tokens * 0.28) / 1_000_000
        self.total_cost_usd += cost

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "model": self.model,
            "finish_reason": response.choices[0].finish_reason
        }

    def propose_action(self,
                      task_description: str,
                      context: Dict[str, Any],
                      action_schema: Dict[str, Any]) -> Dict[str, Any]:
        """让模型提议一个行动。

        Args:
            task_description: 任务描述
            context: 上下文信息（当前状态、历史等）
            action_schema: 行动的 JSON schema

        Returns:
            提议的行动（已验证 schema）
        """
        system_prompt = f"""You are an AI assistant for continuous sign language recognition experiments.

Your task: {task_description}

You can propose actions by generating JSON that follows this schema:
{json.dumps(action_schema, indent=2)}

Guidelines:
1. Only propose valid actions according to the schema
2. Use information from the context to make informed decisions
3. Explain your reasoning briefly in a 'rationale' field
4. Be conservative: prefer simple, safe actions

Context:
{json.dumps(context, indent=2)}

Respond with a JSON object containing your proposed action.
"""

        user_message = "Based on the context, propose the next action to take."

        response = self.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            response_format="json_object"
        )

        return response

    def diagnose_error(self, error_log: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """诊断错误（使用 Reasoner 模型）。

        Args:
            error_log: 错误日志
            context: 上下文信息

        Returns:
            诊断结果和建议
        """
        # 切换到 reasoner 模型（如果需要）
        original_model = self.model
        if "reasoner" not in self.model:
            self.model = "deepseek-reasoner"

        system_prompt = """You are an expert at debugging continuous sign language recognition experiments.

Given an error log and context, analyze the problem and suggest fixes.

Provide:
1. Root cause analysis
2. Specific fix suggestions
3. Whether to retry or skip this experiment
"""

        user_message = f"""Error log:
{error_log}

Context:
{json.dumps(context, indent=2)}

Please diagnose the error and provide actionable suggestions.
"""

        response = self.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            response_format="json_object"
        )

        # 恢复原始模型
        self.model = original_model

        return response

    def reflect_on_progress(self,
                           experiment_history: List[Dict[str, Any]],
                           goal: str) -> Dict[str, Any]:
        """反思实验进展，调整策略。

        Args:
            experiment_history: 实验历史记录
            goal: 目标描述

        Returns:
            反思结果和策略调整建议
        """
        system_prompt = f"""You are an AI assistant conducting sign language recognition experiments.

Goal: {goal}

Review the experiment history and reflect:
1. What patterns do you see in the results?
2. Are you making progress toward the goal?
3. Should you adjust your strategy?
4. What should you try next?

Be data-driven and specific.
"""

        user_message = f"""Experiment history:
{json.dumps(experiment_history, indent=2)}

Please reflect on the progress and suggest next steps.
"""

        response = self.chat(
            system_prompt=system_prompt,
            user_message=user_message,
            response_format="json_object"
        )

        return response

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        Returns:
            API 调用统计
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4)
        }

    def reset_stats(self):
        """重置统计信息。"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0


# 示例用法
if __name__ == "__main__":
    # 需要设置环境变量 DEEPSEEK_API_KEY
    try:
        client = DeepSeekClient(model="deepseek-chat")

        # 简单聊天测试
        response = client.chat(
            system_prompt="You are a helpful assistant.",
            user_message="What is continuous sign language recognition?",
        )

        print("Response:", response["content"])
        print(f"Cost: ${response['cost_usd']:.6f}")
        print(f"Tokens: {response['input_tokens']} in + {response['output_tokens']} out")

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set DEEPSEEK_API_KEY environment variable")
