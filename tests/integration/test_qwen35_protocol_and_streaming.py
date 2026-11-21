import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.atlas.server as s


class _CharStreamBackend:
    mtp = "off"

    def __init__(self, text: str):
        self.text = text
        self.prompts = []

    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass

    async def generate(self, prompt, n_predict=512, temp=0.7, top_p=0.9, min_p=0.0, stop=None, req_id=None, mtp_draft_n=None):
        self.prompts.append(prompt)
        for ch in self.text:
            yield {"event": "token", "token": ch, "n_tokens": 1}
        yield {"event": "done", "prompt_tokens": 50, "completion_tokens": len(self.text), "finish_reason": "stop"}


def test_parse_tools_from_prompt_supports_jsonlines_and_array():
    jsonl_prompt = (
        '<tools>\n{"type": "function", "function": {"name": "Read"}}\n'
        '{"type": "function", "function": {"name": "Write"}}\n</tools>'
    )
    arr_prompt = '<tools>\n[{"type": "function", "function": {"name": "Bash"}}]\n</tools>'
    assert [t["function"]["name"] for t in s.parse_tools_from_prompt(jsonl_prompt)] == ["Read", "Write"]
    assert [t["function"]["name"] for t in s.parse_tools_from_prompt(arr_prompt)] == ["Bash"]


def test_format_chatml_prompt_coalesces_consecutive_tool_turns():
    msgs = [
        s.ChatMessage(role="user", content="Read two files"),
        s.ChatMessage(
            role="assistant",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": '{"path": "a.py"}'}},
                {"id": "c2", "type": "function", "function": {"name": "Read", "arguments": '{"path": "b.py"}'}},
            ],
        ),
        s.ChatMessage(role="tool", tool_call_id="c1", content="alpha"),
        s.ChatMessage(role="tool", tool_call_id="c2", content="beta"),
    ]
    tools = [s.ToolDefinition(type="function", function=s.FunctionDefinition(name="Read", parameters={"type": "object"}))]
    prompt = s.format_chatml_prompt(msgs, tools=tools, enable_thinking=True, tool_format="qwen35_xml")
    expected_block = "<|im_start|>user\n<tool_response>\nalpha\n</tool_response>\n<tool_response>\nbeta\n</tool_response><|im_end|>"
    assert expected_block in prompt


def test_openai_streaming_qwen35_xml_thinking_and_split_tags():
    payload_text = (
        "Think step 1</think>\n\nChecking files.<response_to_instruction>\n"
        "<tool_call>\n<function=Write>\n"
        "<parameter=file_path>\napp.py\n</parameter>\n"
        "<parameter=content>\nprint(1 < 2)\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    backend = _CharStreamBackend(payload_text)
    app = s.create_app(backend=backend)
    client = TestClient(app)

    r = client.post("/v1/chat/completions", json={
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": "Create app.py"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
                },
            },
        }],
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "stream": True,
    })
    assert r.status_code == 200

    reasoning = []
    content = []
    tool_names = []
    tool_args = []
    finish_reasons = []
    for line in r.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        c0 = chunk["choices"][0]
        if c0.get("finish_reason"):
            finish_reasons.append(c0["finish_reason"])
        d = c0.get("delta", {})
        if d.get("reasoning_content"):
            reasoning.append(d["reasoning_content"])
        if d.get("content"):
            content.append(d["content"])
        for tc in d.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name"):
                tool_names.append(fn["name"])
            if fn.get("arguments"):
                tool_args.append(fn["arguments"])

    assert "".join(reasoning) == "Think step 1"
    assert "".join(content).strip() == "Checking files."
    assert tool_names == ["Write"]
    parsed = json.loads("".join(tool_args))
    assert parsed == {"file_path": "app.py", "content": "print(1 < 2)"}
    assert finish_reasons[-1] == "tool_calls"


def test_openai_responses_endpoint_forwards_text_format_and_reasoning():
    backend = _CharStreamBackend('```json\n{"ok": true}\n```')
    app = s.create_app(backend=backend)
    client = TestClient(app)

    r = client.post("/v1/responses", json={
        "model": "qwen3.8-flash-next",
        "input": "Return ok",
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}},
        "stream": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == '{"ok": true}'
    assert "Respond with a valid JSON object only." in backend.prompts[0]
