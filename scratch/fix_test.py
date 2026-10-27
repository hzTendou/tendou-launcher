with open("tests/integration/test_openai_server.py", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """        custom_app = create_app(backend=AngleBracketBackend())
        transport = httpx.ASGITransport(app=custom_app)
        with httpx.Client(transport=transport, base_url="http://test") as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "atlas-engine",
                    "messages": [{"role": "user", "content": "Write template header"}],
                    "tools": [{"type": "function", "function": {"name": "dummy_tool", "parameters": {}}}],
                    "stream": True,
                },
            )
            self.assertEqual(resp.status_code, 200)
            accumulated = ""
            for line in resp.iter_lines():
                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"]
                    if delta.get("content"):
                        accumulated += delta["content"]
            self.assertEqual(accumulated, "template <")"""

new_block = """        async def run_test():
            custom_app = create_app(backend=AngleBracketBackend())
            transport = httpx.ASGITransport(app=custom_app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "atlas-engine",
                        "messages": [{"role": "user", "content": "Write template header"}],
                        "tools": [{"type": "function", "function": {"name": "dummy_tool", "parameters": {}}}],
                        "stream": True,
                    },
                ) as resp:
                    self.assertEqual(resp.status_code, 200)
                    accumulated = ""
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and not line.endswith("[DONE]"):
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"]
                            if delta.get("content"):
                                accumulated += delta["content"]
                    self.assertEqual(accumulated, "template <")

        asyncio.run(run_test())"""

assert old_block in text, "old_block not found!"
text = text.replace(old_block, new_block, 1)

with open("tests/integration/test_openai_server.py", "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: fixed async test in test_openai_server.py")
