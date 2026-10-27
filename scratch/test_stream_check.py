import sys
sys.path.insert(0, ".")
import asyncio
import json
from src.atlas.server import create_app, MockAtlasBackend
from httpx import AsyncClient, ASGITransport

async def run():
    app = create_app(backend=MockAtlasBackend())
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as c:
        r = await c.post('/v1/chat/completions', json={'model': 'test', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True})
        for line in r.text.split('\n'):
            if line.startswith('data: ') and not line.endswith('[DONE]'):
                d = json.loads(line[6:])
                if d['choices'][0]['delta'].get('content'):
                    print('CHUNK:', repr(d['choices'][0]['delta']['content']))

asyncio.run(run())
