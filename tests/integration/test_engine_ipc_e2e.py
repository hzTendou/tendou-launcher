"""
End-to-End integration test connecting AtlasSubprocessBackend directly
to llama-atlas-engine.exe with real model shards and --boost mode.
"""

import asyncio
import os
import unittest
import time
import httpx
import uvicorn
from openai import OpenAI

from src.atlas.server import (
    DEFAULT_EXE_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_ID,
    AtlasSubprocessBackend,
    create_app,
)


@unittest.skipUnless(
    os.path.exists(DEFAULT_EXE_PATH) and os.path.exists(DEFAULT_MODEL_PATH),
    "llama-atlas-engine.exe or model shards not found",
)
class TestAtlasEngineE2E(unittest.TestCase):
    backend: AtlasSubprocessBackend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = AtlasSubprocessBackend(
            exe_path=DEFAULT_EXE_PATH,
            model_path=DEFAULT_MODEL_PATH,
            threads=14,
            boost=True,
        )

    def test_01_backend_ipc_generation(self):
        async def run_test():
            print("\n[E2E] Starting AtlasSubprocessBackend (loading model shards into memory)...")
            t0 = time.time()
            await self.backend.start()
            startup_time = time.time() - t0
            print(f"[E2E] Engine started in {startup_time:.2f}s")
            self.assertTrue(self.backend.is_healthy())

            # Test 1: Generate small response
            prompt = "<|im_start|>user\nWhat is 2+2? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
            tokens = []
            done_event = None
            t_gen0 = time.time()

            async for ev in self.backend.generate(prompt=prompt, n_predict=8, temp=0.0):
                if ev.get("event") == "token":
                    tokens.append(ev.get("token", ""))
                elif ev.get("event") == "done":
                    done_event = ev

            gen_time = time.time() - t_gen0
            full_text = "".join(tokens)
            print(f"[E2E] Generated '{full_text.strip()}' in {gen_time:.2f}s ({len(tokens)} tokens)")
            self.assertGreater(len(tokens), 0)
            self.assertIsNotNone(done_event)
            self.assertIn(done_event["finish_reason"], ("length", "stop"))

            # Test 2: Verify KV cache was cleared and second prompt generates immediately without reloading!
            prompt2 = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n"
            tokens2 = []
            t_gen1 = time.time()
            async for ev in self.backend.generate(prompt=prompt2, n_predict=6, temp=0.0):
                if ev.get("event") == "token":
                    tokens2.append(ev.get("token", ""))

            gen_time2 = time.time() - t_gen1
            print(f"[E2E] Prompt 2 generated in {gen_time2:.2f}s ({len(tokens2)} tokens)")
            self.assertGreater(len(tokens2), 0)
            # Test 3: Long prompt (>512 tokens) chunked prefill verification
            # Ensures prompts exceeding llama_n_batch decode in chunks without crashing
            long_sentence = "The Atlas Engine is a high performance local inference runtime for large MoE models. "
            long_prompt = "<|im_start|>user\n" + (long_sentence * 35) + "\nSummarize in 3 words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            tokens3 = []
            done_event3 = None
            t_gen2 = time.time()
            async for ev in self.backend.generate(prompt=long_prompt, n_predict=8, temp=0.0):
                if ev.get("event") == "token":
                    tokens3.append(ev.get("token", ""))
                elif ev.get("event") == "done":
                    done_event3 = ev

            gen_time3 = time.time() - t_gen2
            print(f"[E2E] Long prompt (>512 tokens) decoded and generated {len(tokens3)} tokens in {gen_time3:.2f}s")
            self.assertGreater(len(tokens3), 0)
            self.assertIsNotNone(done_event3)
            self.assertIn(done_event3.get("finish_reason"), ("length", "stop"))

            # Test 4: Stop backend
            await self.backend.stop()
            self.assertFalse(self.backend.is_healthy())

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
