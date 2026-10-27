with open("tests/integration/test_openai_server.py", "r", encoding="utf-8") as f:
    text = f.read()

new_tests = '''    # -----------------------------------------------------------------------
    # 16. Streaming Quality & Latency Tests (Phase 11.5 / 12 Verification)
    # -----------------------------------------------------------------------

    def test_no_duplicate_content_in_streaming(self):
        """Verify that streaming chat completion does not duplicate content at stream termination."""
        stream = self.client.chat.completions.create(
            model=DEFAULT_MODEL_ID,
            messages=[{"role": "user", "content": "Tell me about memory hierarchy"}],
            stream=True,
            temperature=0.0,
            max_tokens=128,
        )
        content_pieces = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_pieces.append(chunk.choices[0].delta.content)

        full_content = "".join(content_pieces)
        self.assertGreater(len(full_content), 0)
        # Verify text is not emitted twice
        target_phrase = "Houses multi-part 134GB GGUF model shards"
        count = full_content.count(target_phrase)
        self.assertEqual(count, 1, f"Expected 1 occurrence of '{target_phrase}', found {count}. Full text:\\n{full_content}")

    def test_tag_buffer_angle_bracket_preservation(self):
        """Verify that when tools are enabled, trailing angle brackets like '<' in C++ or HTML are preserved and not swallowed."""
        from src.atlas.server import BaseAtlasBackend
        from typing import AsyncIterator

        class AngleBracketBackend(BaseAtlasBackend):
            def __init__(self):
                self._is_ready = True
            async def start(self): pass
            async def stop(self): pass
            async def generate(self, prompt, **kwargs) -> AsyncIterator[Dict[str, Any]]:
                # Stream code piece by piece, ending with '<' in tag_buffer
                yield {"event": "token", "token": "template ", "n_tokens": 1}
                yield {"event": "token", "token": "<", "n_tokens": 1}
                yield {"event": "done", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 2}
            async def cancel(self, req_id: str): pass
            def is_healthy(self): return True
            def get_model_info(self): return {"id": "test"}

        custom_app = create_app(backend=AngleBracketBackend())
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
            self.assertEqual(accumulated, "template <")

    def test_sse_low_latency_headers(self):
        """Verify that streaming response headers include low-latency SSE buffering directives."""
        with httpx.Client(base_url=self.base_url) as client:
            resp = client.post(
                "/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            self.assertEqual(resp.status_code, 200)
            headers = resp.headers
            self.assertEqual(headers.get("x-accel-buffering"), "no")
            self.assertIn("no-cache", headers.get("cache-control", ""))
            self.assertIn("no-transform", headers.get("cache-control", ""))
            self.assertEqual(headers.get("connection"), "keep-alive")
'''

target = 'if __name__ == "__main__":'
assert target in text, "Target not found!"
text = text.replace(target, new_tests + "\n\n" + target)

with open("tests/integration/test_openai_server.py", "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: added new tests to test_openai_server.py")
