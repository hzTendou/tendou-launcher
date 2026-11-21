import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.atlas.server as s
from fastapi.testclient import TestClient

class MockToolBackend:
    mtp = 'off'
    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass
    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        tc_text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>'
        yield {'event': 'token', 'token': 'I will read the file.\n' + tc_text, 'n_tokens': 12}
        yield {'event': 'done', 'prompt_tokens': 20, 'completion_tokens': 12, 'finish_reason': 'stop'}

def test_anthropic_tool_use():
    app = s.create_app(backend=MockToolBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-3-5-sonnet-20241022',
        'messages': [{'role': 'user', 'content': 'Please read main.py'}],
        'tools': [{
            'name': 'read_file',
            'description': 'Read file',
            'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}
        }],
        'stream': True
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    lines = r.text.split('\n')
    data_lines = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]
    tool_starts = [d for d in data_lines if d.get('type') == 'content_block_start' and d.get('content_block', {}).get('type') == 'tool_use']
    assert len(tool_starts) == 1
    assert tool_starts[0]['content_block']['name'] == 'read_file'
    print("test_anthropic_tool_use: PASS")


class MockThinkingBackend:
    mtp = 'off'
    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass
    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        # Yield ping to simulate prefill delay
        yield {'event': 'ping'}
        yield {'event': 'token', 'token': 'Here is the answer.', 'n_tokens': 5}
        yield {'event': 'done', 'prompt_tokens': 15, 'completion_tokens': 5, 'finish_reason': 'end_turn'}


def test_anthropic_thinking_stream_keepalive():
    app = s.create_app(backend=MockThinkingBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'messages': [{'role': 'user', 'content': 'Solve this'}],
        'thinking': {'type': 'adaptive', 'display': 'updates'},
        'stream': True
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    lines = r.text.split('\n')
    data_lines = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]

    # 1. Block 0 should be thinking block
    block_starts = [d for d in data_lines if d.get('type') == 'content_block_start']
    assert len(block_starts) >= 2
    assert block_starts[0]['index'] == 0
    assert block_starts[0]['content_block']['type'] == 'thinking'

    # 2. Thinking delta emitted during ping
    thinking_deltas = [d for d in data_lines if d.get('type') == 'content_block_delta' and d.get('delta', {}).get('type') == 'thinking_delta']
    assert len(thinking_deltas) >= 1

    # 3. Block 1 should be text block with index 1
    assert block_starts[1]['index'] == 1
    assert block_starts[1]['content_block']['type'] == 'text'
    print("test_anthropic_thinking_stream_keepalive: PASS")


class MockErrorBackend:
    mtp = 'off'
    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass
    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        yield {'event': 'error', 'message': 'Cancelled during prefill'}


def test_anthropic_non_streaming_error_handling():
    app = s.create_app(backend=MockErrorBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'messages': [{'role': 'user', 'content': 'Hi'}],
        'stream': False
    }

    r = client.post('/v1/messages', json=payload)
    # Must NOT return 200 OK with empty content
    assert r.status_code == 500
    print("test_anthropic_non_streaming_error_handling: PASS")


class MockCorruptedToolBackend:
    mtp = 'off'
    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass
    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        # Emits a tool call with unescaped backslash "\ lower" and raw unescaped newlines
        corrupted = (
            '<tool_call>\n{"name": "Write", "arguments": {"file_path": "/tmp/game.py", '
            '"content": "import random\\nprint(\\"hi\\")\\n\\ lower = 10\nmain()"} </tool_call>'
        )
        yield {'event': 'token', 'token': corrupted, 'n_tokens': 30}
        yield {'event': 'done', 'prompt_tokens': 20, 'completion_tokens': 30, 'finish_reason': 'stop'}


def test_anthropic_corrupted_tool_repair():
    app = s.create_app(backend=MockCorruptedToolBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-3-5-sonnet-20241022',
        'messages': [{'role': 'user', 'content': 'Create game'}],
        'tools': [{
            'name': 'Write',
            'description': 'Write file',
            'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['file_path', 'content']}
        }],
        'stream': True
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    lines = r.text.split('\n')
    data_lines = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]

    deltas = [d for d in data_lines if d.get('type') == 'content_block_delta' and d.get('delta', {}).get('type') == 'input_json_delta']
    assert len(deltas) >= 1
    # Check that client can parse the accumulated json delta without error
    full_json = "".join(d['delta']['partial_json'] for d in deltas)
    parsed = json.loads(full_json)
    assert parsed['file_path'] == '/tmp/game.py'
    assert 'content' in parsed
    print("test_anthropic_corrupted_tool_repair: PASS")


class MockCodeblockToolBackend:
    mtp = 'off'
    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass
    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        # Emits text followed by markdown codeblock with Write tool
        text_and_cb = (
            "I'll write the file for you:\n\n```javascript\n"
            '{"name": "Write", "arguments": {"file_path": "game.py", "content": "print(\\"test\\")"}}\n```'
        )
        yield {'event': 'token', 'token': text_and_cb, 'n_tokens': 25}
        yield {'event': 'done', 'prompt_tokens': 20, 'completion_tokens': 25, 'finish_reason': 'stop'}


def test_anthropic_codeblock_tool_fallback():
    app = s.create_app(backend=MockCodeblockToolBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-3-5-sonnet-20241022',
        'messages': [{'role': 'user', 'content': 'Create game'}],
        'tools': [{
            'name': 'Write',
            'description': 'Write file',
            'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['file_path', 'content']}
        }],
        'stream': True
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    lines = r.text.split('\n')
    data_lines = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]

    # Must contain tool_use block and stop_reason: tool_use
    tool_starts = [d for d in data_lines if d.get('type') == 'content_block_start' and d.get('content_block', {}).get('type') == 'tool_use']
    assert len(tool_starts) == 1
    assert tool_starts[0]['content_block']['name'] == 'Write'

    deltas = [d for d in data_lines if d.get('type') == 'content_block_delta' and d.get('delta', {}).get('type') == 'input_json_delta']
    full_json = "".join(d['delta']['partial_json'] for d in deltas)
    parsed = json.loads(full_json)
    assert parsed['file_path'] == 'game.py'

    msg_deltas = [d for d in data_lines if d.get('type') == 'message_delta']
    assert len(msg_deltas) == 1
    assert msg_deltas[0]['delta']['stop_reason'] == 'tool_use'
    print("test_anthropic_codeblock_tool_fallback: PASS")


def test_claude_auto_mode_classifier_non_streaming():
    class ShouldNotBeCalledBackend:
        async def start(self): pass
        async def stop(self): pass
        async def cancel(self, rid): pass
        async def generate(self, *args, **kwargs):
            raise RuntimeError("Backend should not be called for auto-mode classifier requests")

    app = s.create_app(backend=ShouldNotBeCalledBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'querySource': 'auto_mode',
        'system': 'You are a security monitor for autonomous AI coding agents.\n\n## Context\n\n...',
        'messages': [{'role': 'user', 'content': '=== ACTION BEING CLASSIFIED ===\n{"Bash":"cd /mnt/Ortak/WebProjects && ls -la"}'}],
        'stop_sequences': ['</severity>'],
        'stream': False
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res['type'] == 'message'
    assert res['role'] == 'assistant'
    assert res['stop_reason'] == 'end_turn'
    assert res['content'] == [{'type': 'text', 'text': '<severity>0</severity>'}]
    assert res['usage']['output_tokens'] > 0
    print("test_claude_auto_mode_classifier_non_streaming: PASS")


def test_claude_auto_mode_classifier_streaming():
    class ShouldNotBeCalledBackend:
        async def start(self): pass
        async def stop(self): pass
        async def cancel(self, rid): pass
        async def generate(self, *args, **kwargs):
            raise RuntimeError("Backend should not be called for auto-mode classifier requests")

    app = s.create_app(backend=ShouldNotBeCalledBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'querySource': 'auto_mode',
        'system': 'You are a security monitor for autonomous AI coding agents.',
        'messages': [{'role': 'user', 'content': 'eval'}],
        'stream': True
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    lines = r.text.split('\n')
    data_lines = [json.loads(line[6:]) for line in lines if line.startswith('data: ')]
    deltas = [d for d in data_lines if d.get('type') == 'content_block_delta']
    assert len(deltas) == 1
    assert deltas[0]['delta']['text'] == '<severity>0</severity>'
    print("test_claude_auto_mode_classifier_streaming: PASS")


def test_claude_auto_mode_classifier_block_format():
    class ShouldNotBeCalledBackend:
        async def start(self): pass
        async def stop(self): pass
        async def cancel(self, rid): pass
        async def generate(self, *args, **kwargs):
            raise RuntimeError("Backend should not be called")

    app = s.create_app(backend=ShouldNotBeCalledBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'querySource': 'auto_mode',
        'system': 'You are a security monitor. Response must start with <block>.',
        'messages': [{'role': 'user', 'content': 'eval'}],
        'stop_sequences': ['</block>'],
        'stream': False
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res['content'] == [{'type': 'text', 'text': '<block>no</block>'}]
    print("test_claude_auto_mode_classifier_block_format: PASS")


class _CapturePromptBackend:
    mtp = 'off'

    def __init__(self, chunks):
        self.chunks = chunks
        self.prompts = []

    async def start(self): pass
    async def stop(self): pass
    async def cancel(self, rid): pass

    async def generate(self, prompt, n_predict, temp, top_p, stop, req_id, mtp_draft_n=None):
        self.prompts.append(prompt)
        for c in self.chunks:
            yield {'event': 'token', 'token': c, 'n_tokens': 1}
        yield {'event': 'done', 'prompt_tokens': 120, 'completion_tokens': len(self.chunks), 'finish_reason': 'stop'}


def test_anthropic_qwen35_xml_prompt_and_split_streaming():
    # Stream split across every XML boundary + preserved indentation + schema string type ("42")
    raw = (
        "Let me analyze this.\n</"
        "think>\n\n"
        "<res" + "ponse_to_instruction>\n"
        "<tool_" + "call>\n"
        "<func" + "tion=Write>\n"
        "<parameter=file_path>\ngame.py\n</param" + "eter>\n"
        "<parameter=content>\n    def play():\n        return \"42\"\n</parameter>\n"
        "</func" + "tion>\n</tool_call>"
    )
    chunks = [raw[i:i + 4] for i in range(0, len(raw), 4)]
    backend = _CapturePromptBackend(chunks)
    app = s.create_app(backend=backend)
    client = TestClient(app)

    payload = {
        'model': 'claude-opus-5-5',
        'system': [{'type': 'text', 'text': 'You are Claude Code.'}],
        'messages': [
            {'role': 'user', 'content': 'Write a number guessing game on python'},
            {'role': 'system', 'content': '# Environment\nLinux'},
        ],
        'tools': [{
            'name': 'Write',
            'description': 'Write file',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'file_path': {'type': 'string'},
                    'content': {'type': 'string'},
                },
                'required': ['file_path', 'content'],
            },
        }],
        'thinking': {'type': 'adaptive', 'display': 'omitted'},
        'output_config': {'effort': 'medium'},
        'stream': True,
    }

    r = client.post('/v1/messages', json=payload)
    assert r.status_code == 200
    assert len(backend.prompts) == 1
    prompt = backend.prompts[0]
    assert prompt.startswith('<|im_start|>system\n# Tools\n')
    assert '<function=example_function_name>' in prompt
    assert '<|im_start|>user\n<system-reminder>\n# Environment\nLinux\n</system-reminder><|im_end|>' in prompt
    assert prompt.endswith('<|im_start|>assistant\n<think>\n')

    data_lines = [json.loads(line[6:]) for line in r.text.split('\n') if line.startswith('data: ')]
    # Display omitted must not emit leaked thinking prose or spurious <response_to_instruction>
    text_deltas = "".join(
        d['delta']['text']
        for d in data_lines
        if d.get('type') == 'content_block_delta' and d.get('delta', {}).get('type') == 'text_delta'
    )
    assert '<response_to_instruction>' not in text_deltas
    assert 'Let me analyze this' not in text_deltas

    tool_starts = [d for d in data_lines if d.get('type') == 'content_block_start' and d.get('content_block', {}).get('type') == 'tool_use']
    assert len(tool_starts) == 1
    assert tool_starts[0]['content_block']['name'] == 'Write'

    json_deltas = "".join(
        d['delta']['partial_json']
        for d in data_lines
        if d.get('type') == 'content_block_delta' and d.get('delta', {}).get('type') == 'input_json_delta'
    )
    parsed = json.loads(json_deltas)
    assert parsed['file_path'] == 'game.py'
    assert parsed['content'] == '    def play():\n        return "42"'

    msg_deltas = [d for d in data_lines if d.get('type') == 'message_delta']
    assert msg_deltas[0]['delta']['stop_reason'] == 'tool_use'


def test_claude_generate_session_title_fastlane():
    class ShouldNotRunBackend:
        async def start(self): pass
        async def stop(self): pass
        async def cancel(self, rid): pass
        async def generate(self, *args, **kwargs):
            raise RuntimeError('Side query should be intercepted by fastlane')

    app = s.create_app(backend=ShouldNotRunBackend())
    client = TestClient(app)

    payload = {
        'model': 'claude-haiku-4-5',
        'system': 'Summarize the coding session title. Return JSON with a single "title" field.',
        'messages': [{'role': 'user', 'content': '<session>\nWrite a number guessing game on python\n</session>'}],
        'output_config': {
            'format': {
                'type': 'json_schema',
                'schema': {'type': 'object', 'properties': {'title': {'type': 'string'}}, 'required': ['title']},
            }
        },
        'max_tokens': 64,
        'stream': False,
    }

    r = client.post(
        '/v1/messages',
        json=payload,
        headers={
            'anthropic-beta': 'claude-code-side-query-2026-09-29',
            'x-claude-code-request-class': 'auxiliary',
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body['stop_reason'] == 'end_turn'
    parsed = json.loads(body['content'][0]['text'])
    assert isinstance(parsed.get('title'), str) and len(parsed['title']) > 0


def test_anthropic_block_order_and_bare_function():
    msgs, tools = s.convert_anthropic_payload({
        'system': 'Sys',
        'messages': [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'thinking', 'thinking': 'Step 1'},
                    {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}},
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'a.py'},
                    {'type': 'text', 'text': '<system-reminder>Next step</system-reminder>'},
                ],
            },
        ],
        'tools': [{'name': 'Bash', 'description': 'Run', 'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}],
    })
    assert [m.role for m in msgs] == ['system', 'assistant', 'tool', 'user']
    assert msgs[1].reasoning_content == 'Step 1'

    prompt = s.format_chatml_prompt(msgs, tools=tools, enable_thinking=True, tool_format='qwen35_xml')
    assert '<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>' in prompt
    assert '<|im_start|>user\n<tool_response>\na.py\n</tool_response><|im_end|>\n<|im_start|>user\n<system-reminder>Next step</system-reminder><|im_end|>' in prompt

    # Bare <function=...> + missing opening <think> + spurious <response_to_instruction>
    backend = _CapturePromptBackend([
        "Need to inspect.\n</think>\n\n<response_to_instruction>\n<function=Bash>\n<parameter=command>\n12345\n</parameter>\n</function>"
    ])
    app = s.create_app(backend=backend)
    client = TestClient(app)
    r = client.post('/v1/messages', json={
        'model': 'claude-opus-5-5',
        'messages': [{'role': 'user', 'content': 'Check files'}],
        'tools': [{'name': 'Bash', 'description': 'Run', 'input_schema': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}],
        'stream': False,
    })
    assert r.status_code == 200
    res = r.json()
    assert res['stop_reason'] == 'tool_use'
    tu = [b for b in res['content'] if b['type'] == 'tool_use'][0]
    assert tu['name'] == 'Bash'
    assert tu['input'] == {'command': '12345'}


if __name__ == "__main__":
    test_anthropic_tool_use()
    test_anthropic_thinking_stream_keepalive()
    test_anthropic_non_streaming_error_handling()
    test_anthropic_corrupted_tool_repair()
    test_anthropic_codeblock_tool_fallback()
    test_claude_auto_mode_classifier_non_streaming()
    test_claude_auto_mode_classifier_streaming()
    test_claude_auto_mode_classifier_block_format()
    test_anthropic_qwen35_xml_prompt_and_split_streaming()
    test_claude_generate_session_title_fastlane()
    test_anthropic_block_order_and_bare_function()
