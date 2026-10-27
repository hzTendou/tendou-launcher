"""Integration tests for Cline (NVIDIA NIM / Agentic Coding XML & OpenAI/Anthropic Protocol) support."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.atlas.server as s
from src.atlas.autotune import estimate_prompt_pld_suitability


CLINE_SYSTEM_PROMPT = (
    "You are Cline, a highly skilled software engineer with extensive knowledge in many programming "
    "languages, frameworks, design patterns, and best practices.\n\n"
    "====\n\n"
    "TOOL USE\n\n"
    "You have access to a set of tools that are executed upon the user's approval. You can use one tool per message, "
    "and will receive the result of that tool use in the user's response.\n\n"
    "# Tools\n"
    "## read_file\nDescription: Request to read the contents of a file.\n"
    "Usage:\n<read_file>\n<path>File path here</path>\n</read_file>\n\n"
    "## replace_in_file\nDescription: Request to replace sections of content in an existing file using SEARCH/REPLACE blocks.\n"
    "Usage:\n<replace_in_file>\n<path>File path here</path>\n<diff>\n<<<<<<< SEARCH\n[exact content]\n=======\n[new content]\n>>>>>>> REPLACE\n</diff>\n</replace_in_file>\n\n"
    "## execute_command\nDescription: Request to execute a CLI command.\n"
    "Usage:\n<execute_command>\n<command>CLI command</command>\n<requires_approval>false</requires_approval>\n</execute_command>\n\n"
    "## attempt_completion\nDescription: Present the result of your work.\n"
    "Usage:\n<attempt_completion>\n<result>Final result</result>\n</attempt_completion>"
)


def test_cline_multipart_content_flattening_and_prefix_determinism():
    """Verifies that Cline's array-based content blocks are cleanly flattened without JSON bracket pollution."""
    msgs = [
        s.ChatMessage(
            role="system",
            content=[{"type": "text", "text": CLINE_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        ),
        s.ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "<task>\nInspect src/atlas/server.py and fix the handler.\n</task>"},
                {
                    "type": "text",
                    "text": "<environment_details>\n# VSCode Visible Files\nsrc/atlas/server.py\n</environment_details>",
                },
            ],
        ),
    ]
    prompt = s.format_chatml_prompt(msgs)
    assert '[{"type":' not in prompt
    assert "cache_control" not in prompt
    assert "<|im_start|>system\nYou are Cline" in prompt
    assert "<task>\nInspect src/atlas/server.py" in prompt
    assert "<environment_details>\n# VSCode Visible Files" in prompt
    assert s.is_cline_xml_prompt(prompt) is True


def test_cline_pld_suitability_exploration_vs_replace_in_file():
    """Verifies that Cline exploration uses d*=0 while Cline replace_in_file with code context uses PLD d*=3."""
    # 1. Turn 1: Exploration (read_file) -> d*=0
    msgs_explore = [
        s.ChatMessage(role="system", content=CLINE_SYSTEM_PROMPT),
        s.ChatMessage(
            role="user",
            content="<task>\nPlease read src/atlas/server.py\n</task>\n<environment_details>\nsrc/atlas/server.py\n</environment_details>",
        ),
    ]
    prompt_explore = s.format_chatml_prompt(msgs_explore)
    use_pld_1, d_star_1, _ = estimate_prompt_pld_suitability(prompt_explore)
    assert use_pld_1 is False
    assert d_star_1 == 0

    # 2. Turn 2: File read result in context + replace_in_file refactoring request -> PLD d*=3
    msgs_edit = [
        s.ChatMessage(role="system", content=CLINE_SYSTEM_PROMPT),
        s.ChatMessage(role="user", content="<task>\nRead and refactor src/atlas/server.py\n</task>"),
        s.ChatMessage(
            role="assistant",
            content="<thinking>\nReading file.\n</thinking>\n<read_file>\n<path>src/atlas/server.py</path>\n</read_file>",
        ),
        s.ChatMessage(
            role="user",
            content=(
                "[read_file for 'src/atlas/server.py'] Result:\n"
                "def handle_request(req):\n"
                "    if req is None:\n"
                "        return False\n"
                "    return True\n\n"
                "Please refactor handle_request using replace_in_file to optimize validation."
            ),
        ),
    ]
    prompt_edit = s.format_chatml_prompt(msgs_edit)
    use_pld_2, d_star_2, _ = estimate_prompt_pld_suitability(prompt_edit)
    assert use_pld_2 is True
    assert d_star_2 == 3


def test_cline_xml_streaming_and_usage_telemetry():
    """Verifies Cline streaming over /v1/chat/completions with stream_options and multi-turn XML tools."""
    app = s.create_app(backend=s.MockAtlasBackend())
    client = TestClient(app)

    # Turn 1: Cline sends array messages + stream_options={"include_usage": True}
    payload = {
        "model": "qwen3.6-35b",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": CLINE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>\nInspect src/atlas/server.py\n</task>"},
                    {"type": "text", "text": "<environment_details>\nsrc/atlas/server.py\n</environment_details>"},
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Title": "Cline", "HTTP-Referer": "https://cline.bot"},
    )
    assert resp.status_code == 200

    accumulated = ""
    usage_obj = None
    for line in resp.text.splitlines():
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
        chunk = json.loads(data_str)
        if chunk.get("choices"):
            delta = chunk["choices"][0].get("delta", {})
            accumulated += delta.get("content", "")
        if "usage" in chunk:
            usage_obj = chunk["usage"]

    assert "<thinking>" in accumulated
    assert "<read_file>" in accumulated
    assert "<path>src/atlas/server.py</path>" in accumulated
    assert "</read_file>" in accumulated
    assert usage_obj is not None
    assert "prompt_tokens" in usage_obj
    assert "completion_tokens" in usage_obj
    assert "cache_read_input_tokens" in usage_obj
    assert "prompt_tokens_details" in usage_obj


def test_cline_single_tool_truncation_and_qwen_bridge():
    """Verifies that hallucinated trailing tool calls are truncated and accidental <tool_call> JSON is bridged to Cline XML."""

    class HallucinatingClineBackend:
        mtp = "pld"

        async def start(self):
            pass

        async def stop(self):
            pass

        async def cancel(self, rid: str):
            pass

        async def generate(self, prompt, n_predict, temp, top_p, min_p=None, stop=None, req_id=None, mtp_draft_n=None):
            # Emits a valid Cline XML tool call followed by hallucinated fake user environment_details and second tool
            pieces = [
                "<thinking>\nI will read main.py.\n</thinking>\n",
                "<read_file>\n<path>main.py</path>\n</read_file>",
                "\nUser: <environment_details>hallucinated</environment_details>\n<execute_command><command>rm -rf</command></execute_command>",
            ]
            for p in pieces:
                yield {"event": "token", "token": p, "n_tokens": 5}
            yield {
                "event": "done",
                "prompt_tokens": 120,
                "completion_tokens": 15,
                "cached_prompt_tokens": 100,
                "finish_reason": "stop",
            }

    app = s.create_app(backend=HallucinatingClineBackend())
    client = TestClient(app)

    payload = {
        "model": "qwen3.6-35b",
        "messages": [
            {"role": "system", "content": CLINE_SYSTEM_PROMPT},
            {"role": "user", "content": "<task>Read main.py</task>"},
        ],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    streamed_text = ""
    for line in resp.text.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            ch = json.loads(line[6:])
            if ch.get("choices"):
                streamed_text += ch["choices"][0].get("delta", {}).get("content", "")

    assert streamed_text.endswith("</read_file>")
    assert "hallucinated" not in streamed_text
    assert "<execute_command>" not in streamed_text

    # Also verify Qwen <tool_call> JSON -> Cline XML conversion helper
    raw_qwen_tc = '<thinking>Checking file</thinking>\n<tool_call>\n{"name": "read_file", "arguments": {"path": "src/app.py"}}\n</tool_call>'
    bridged = s.convert_qwen_tool_call_to_cline_xml(raw_qwen_tc)
    assert "<read_file>\n<path>src/app.py</path>\n</read_file>" in bridged
    assert "<tool_call>" not in bridged
