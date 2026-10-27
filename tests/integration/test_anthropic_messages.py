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

if __name__ == "__main__":
    test_anthropic_tool_use()
