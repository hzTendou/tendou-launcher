"""
Integration tests for Atlas Engine OpenAI-compatible API server.

Covers:
- Python OpenAI SDK directly via base_url
- Streaming chat completions (SSE)
- Tool/function calling (non-streaming & streaming)
- JSON output mode (response_format={"type": "json_object"})
- OpenCode-compatible requests (tools, roles, streaming deltas)
- Text completions (/v1/completions)
- /health and /v1/models endpoints
- Request cancellation handling
- Concurrent request handling
- Stop sequences and token accounting
"""

import asyncio
import json
import socket
import threading
import time
import unittest
from typing import Any, Dict, List

import httpx
import uvicorn
from openai import OpenAI

from src.atlas.server import (
    DEFAULT_MODEL_ID,
    MockAtlasBackend,
    create_app,
    format_chatml_prompt,
    parse_tool_calls_from_text,
)


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestAtlasOpenAIServer(unittest.TestCase):
    server: uvicorn.Server = None
    server_thread: threading.Thread = None
    port: int = 0
    base_url: str = ""
    client: OpenAI = None

    @classmethod
    def setUpClass(cls):
        cls.port = get_free_port()
        cls.base_url = f"http://127.0.0.1:{cls.port}/v1"

        backend = MockAtlasBackend(model_id=DEFAULT_MODEL_ID)
        app = create_app(backend=backend)

        config = uvicorn.Config(app, host="127.0.0.1", port=cls.port, log_level="warning")
        cls.server = uvicorn.Server(config)
        cls.server_thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.server_thread.start()

        # Wait for server to start
        deadline = time.time() + 10.0
        started = False
        while time.time() < deadline:
            try:
                res = httpx.get(f"http://127.0.0.1:{cls.port}/health", timeout=1.0)
                if res.status_code == 200:
                    started = True
                    break
            except Exception:
                time.sleep(0.1)

        assert started, "Server failed to start within timeout"
        cls.client = OpenAI(base_url=cls.base_url, api_key="dummy-key")

    @classmethod
    def tearDownClass(cls):
        if cls.server:
            cls.server.should_exit = True
        if cls.server_thread:
            cls.server_thread.join(timeout=3.0)

    # -----------------------------------------------------------------------
    # 1. Health & Models Endpoints
    # -----------------------------------------------------------------------

    def test_01_health_endpoint(self):
        res = httpx.get(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model", data)
        self.assertIn("uptime_sec", data)

        res_v1 = httpx.get(f"{self.base_url}/health")
        self.assertEqual(res_v1.status_code, 200)

    def test_02_models_endpoint(self):
        models = self.client.models.list()
        model_ids = [m.id for m in models.data]
        self.assertIn(DEFAULT_MODEL_ID, model_ids)
        self.assertIn("atlas-engine", model_ids)

        model = self.client.models.retrieve(DEFAULT_MODEL_ID)
        self.assertEqual(model.id, DEFAULT_MODEL_ID)

    # -----------------------------------------------------------------------
    # 2. Python OpenAI SDK — Chat Completions
    # -----------------------------------------------------------------------

    def test_03_sdk_chat_completion_non_streaming(self):
        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Hello Atlas Engine!"},
            ],
            stream=False,
            temperature=0.5,
            max_tokens=64,
        )

        self.assertTrue(resp.id.startswith("chatcmpl-"))
        self.assertEqual(resp.object, "chat.completion")
        self.assertIsNotNone(resp.choices[0].message.content)
        self.assertIn(resp.choices[0].finish_reason, ("stop", "length"))
        self.assertGreater(resp.usage.prompt_tokens, 0)
        self.assertGreater(resp.usage.completion_tokens, 0)
        self.assertEqual(
            resp.usage.total_tokens,
            resp.usage.prompt_tokens + resp.usage.completion_tokens,
        )

    # -----------------------------------------------------------------------
    # 3. Streaming (SSE)
    # -----------------------------------------------------------------------

    def test_04_sdk_chat_completion_streaming(self):
        stream = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            stream=True,
            temperature=0.0,
            max_tokens=32,
        )

        chunks = []
        finish_reason = None
        for chunk in stream:
            chunks.append(chunk)
            if chunk.choices and chunk.choices[0].delta.content:
                pass
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].object, "chat.completion.chunk")
        self.assertIn(finish_reason, ("stop", "length"))

    # -----------------------------------------------------------------------
    # 4. Tool / Function Calling
    # -----------------------------------------------------------------------

    def test_05_tool_calling_non_streaming(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
            tools=tools,
            tool_choice="auto",
            stream=False,
        )

        msg = resp.choices[0].message
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertIsNotNone(msg.tool_calls)
        self.assertGreaterEqual(len(msg.tool_calls), 1)

        tool_call = msg.tool_calls[0]
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "get_weather")

        # Arguments must be a valid JSON string per OpenAI specification
        args = json.loads(tool_call.function.arguments)
        self.assertIn("location", args)

    def test_06_tool_calling_streaming(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_database",
                    "description": "Search records",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        stream = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Search for user 42"}],
            tools=tools,
            stream=True,
        )

        tool_call_deltas = []
        finish_reason = None

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                tool_call_deltas.extend(chunk.choices[0].delta.tool_calls)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        self.assertGreaterEqual(len(tool_call_deltas), 1)
        self.assertEqual(finish_reason, "tool_calls")
        self.assertEqual(tool_call_deltas[0].function.name, "search_database")

    # -----------------------------------------------------------------------
    # 5. JSON Mode (response_format)
    # -----------------------------------------------------------------------

    def test_07_response_format_json_mode(self):
        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a JSON generator."},
                {"role": "user", "content": "Return status and items list."},
            ],
            response_format={"type": "json_object"},
            stream=False,
        )

        content = resp.choices[0].message.content
        self.assertIsNotNone(content)
        # Verify valid JSON
        parsed = json.loads(content)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(resp.choices[0].finish_reason, "stop")

    # -----------------------------------------------------------------------
    # 6. OpenCode-Compatible Requests
    # -----------------------------------------------------------------------

    def test_08_opencode_compatible_multi_turn_with_tools(self):
        """
        Simulates OpenCode IDE workflow:
        1. OpenCode sends system prompt + tools list + user request.
        2. Assistant returns tool_calls.
        3. OpenCode executes tool and sends role='tool' response.
        4. Assistant returns final answer.
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

        # Step 1: OpenCode tool calling request
        res1 = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": "atlas-engine",
                "messages": [
                    {"role": "system", "content": "You are an OpenCode coding assistant."},
                    {"role": "user", "content": "Check contents of main.py"},
                ],
                "tools": tools,
                "stream": False,
            },
            headers={"Authorization": "Bearer opencode-key"},
        )
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertEqual(data1["choices"][0]["finish_reason"], "tool_calls")
        tc = data1["choices"][0]["message"]["tool_calls"][0]
        tool_call_id = tc["id"]

        # Step 2: OpenCode returns tool output
        res2 = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": "atlas-engine",
                "messages": [
                    {"role": "system", "content": "You are an OpenCode coding assistant."},
                    {"role": "user", "content": "Check contents of main.py"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "main.py"}'},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": 'print("Hello from main.py")',
                    },
                ],
                "tools": tools,
                "stream": False,
            },
        )
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertIn("choices", data2)

    # -----------------------------------------------------------------------
    # 7. Text Completions (/v1/completions)
    # -----------------------------------------------------------------------

    def test_09_sdk_text_completions(self):
        resp = self.client.completions.create(
            model=DEFAULT_MODEL_ID,
            prompt="def fibonacci(n):",
            max_tokens=32,
            temperature=0.2,
        )

        self.assertEqual(resp.object, "text_completion")
        self.assertGreater(len(resp.choices[0].text), 0)
        self.assertIn(resp.choices[0].finish_reason, ("stop", "length"))
        self.assertGreater(resp.usage.total_tokens, 0)

    # -----------------------------------------------------------------------
    # 8. Stop Sequences
    # -----------------------------------------------------------------------

    def test_10_stop_sequences(self):
        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Tell me about Atlas"}],
            stop=["inference", "high-performance"],
            stream=False,
        )
        self.assertEqual(resp.choices[0].finish_reason, "stop")

    # -----------------------------------------------------------------------
    # 9. Concurrency & Queueing
    # -----------------------------------------------------------------------

    def test_11_concurrent_requests(self):
        async def run_concurrent():
            async with httpx.AsyncClient(base_url=self.base_url) as client:
                reqs = [
                    client.post(
                        "/chat/completions",
                        json={
                            "model": DEFAULT_MODEL_ID,
                            "messages": [{"role": "user", "content": f"Prompt {i}"}],
                            "stream": False,
                            "max_tokens": 16,
                        },
                        timeout=10.0,
                    )
                    for i in range(4)
                ]
                responses = await asyncio.gather(*reqs)
                for r in responses:
                    self.assertEqual(r.status_code, 200)
                    self.assertIn("choices", r.json())

        asyncio.run(run_concurrent())

    # -----------------------------------------------------------------------
    # 10. Request Cancellation
    # -----------------------------------------------------------------------

    def test_12_request_cancellation(self):
        """Verify that dropping a streaming connection triggers backend cancellation without crash."""
        cancelled = False
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Long generation"}],
                    "stream": True,
                    "max_tokens": 1000,
                },
                timeout=5.0,
            ) as response:
                self.assertEqual(response.status_code, 200)
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        # Read 1 chunk and abruptly abort
                        cancelled = True
                        break
        except Exception:
            pass

        self.assertTrue(cancelled)
        # Verify server is still alive and responsive after cancellation
        res = httpx.get(f"http://127.0.0.1:{self.port}/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    # -----------------------------------------------------------------------
    # 11. Multi-Tool Streaming (SSE)
    # -----------------------------------------------------------------------

    def test_13_multi_tool_calling_streaming(self):
        """Verify streaming multiple tool calls in a single response does not drop subsequent tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_user",
                    "description": "Find user ID",
                    "parameters": {"type": "object", "properties": {"user": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Find order ID",
                    "parameters": {"type": "object", "properties": {"order": {"type": "string"}}},
                },
            },
        ]

        stream = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Execute multiple tools: lookup both user and order"}],
            tools=tools,
            stream=True,
        )

        tool_calls = {}
        finish_reason = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                for tc in chunk.choices[0].delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": tc.function.name, "arguments": tc.function.arguments or ""}
                    else:
                        tool_calls[idx]["arguments"] += (tc.function.arguments or "")
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        self.assertEqual(finish_reason, "tool_calls")
        self.assertEqual(len(tool_calls), 2, "Both tool calls must be present in streaming mode")
        self.assertEqual(tool_calls[0]["name"], "lookup_user")
        self.assertEqual(tool_calls[1]["name"], "lookup_order")

    # -----------------------------------------------------------------------
    # 12. Qwen XML Tool Calling & Truncation Parsing
    # -----------------------------------------------------------------------

    def test_14_qwen_native_xml_and_unclosed_tool_call_parsing(self):
        """Verify XML format <function=...><parameter=...> and unclosed tool call parsing."""
        # 1. Native Qwen XML format
        xml_text = (
            "I will check the weather for you.\n"
            "<tool_call>\n"
            "<function=get_current_weather>\n"
            "<parameter=location>\n"
            "Tokyo, Japan\n"
            "</parameter>\n"
            "<parameter=unit>\n"
            "celsius\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        content, tcs, reasoning = parse_tool_calls_from_text(xml_text)
        self.assertEqual(content, "I will check the weather for you.")
        self.assertIsNotNone(tcs)
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["function"]["name"], "get_current_weather")
        args = json.loads(tcs[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Tokyo, Japan")
        self.assertEqual(args["unit"], "celsius")

        # 2. Unclosed tool call (cutoff by token limit)
        unclosed_text = '<tool_call>\n{"name": "fetch_file", "arguments": {"path": "test.txt"}}'
        _, unclosed_tcs, _ = parse_tool_calls_from_text(unclosed_text)
        self.assertIsNotNone(unclosed_tcs)
        self.assertEqual(unclosed_tcs[0]["function"]["name"], "fetch_file")

    # -----------------------------------------------------------------------
    # 13. Assistant Message Content & Tool Call History Preservation
    # -----------------------------------------------------------------------

    def test_15_assistant_message_content_and_tool_calls_preservation(self):
        """Verify assistant message having both content and tool_calls preserves content in ChatML prompt."""
        from src.atlas.server import ChatMessage
        messages = [
            ChatMessage(role="user", content="Read file foo.py"),
            ChatMessage(
                role="assistant",
                content="Sure, I will read the file foo.py for you right now.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "foo.py"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", content='print("hello world")', tool_call_id="call_123"),
        ]
        prompt = format_chatml_prompt(messages)
        self.assertIn("Sure, I will read the file foo.py for you right now.", prompt)
        self.assertIn('<tool_call>\n{"name": "read_file", "arguments": {"path": "foo.py"}}', prompt)
        self.assertIn("<tool_response>\nprint(\"hello world\")\n</tool_response>", prompt)

    # -----------------------------------------------------------------------
    # 14. Structured JSON Mode with JSON Schema
    # -----------------------------------------------------------------------

    def test_16_response_format_json_schema(self):
        """Verify structured JSON schema injection and response format."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        resp = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Generate person info"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": schema},
            },
            stream=False,
        )
        content = resp.choices[0].message.content
        self.assertIsNotNone(content)
        parsed = json.loads(content)
        self.assertIsInstance(parsed, dict)

    # -----------------------------------------------------------------------
    # 15. OpenCode Multi-Turn Final Answer
    # -----------------------------------------------------------------------

    def test_17_opencode_multi_turn_completes_after_tool_response(self):
        """Verify that after receiving a tool response, assistant returns a text completion, not a tool call."""
        res = httpx.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": "atlas-engine",
                "messages": [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Check config.json"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": '{"port": 8080}'},
                ],
                "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
                "stream": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNotNone(choice["message"]["content"])
        self.assertIn("verified", choice["message"]["content"].lower())


if __name__ == "__main__":
    unittest.main()
