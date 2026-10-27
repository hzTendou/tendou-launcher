import re

with open("src/atlas/server.py", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Micro-batch queue drain
old_drain = """                    yield event
                    if event.get("event") in ("done", "error"):
                        break"""

new_drain = """                    # Micro-Batch Queue Drain: PLD çoklu token ürettiğinde kuyrukta bekleyen
                    # ardışık tokenları tek event-loop adımında birleştirerek IPC gecikmesini sıfırla
                    if isinstance(event, dict) and event.get("event") == "token":
                        merged_text = event.get("token", "")
                        merged_count = int(event.get("n_tokens", 1))
                        leftover = None
                        while not q.empty():
                            try:
                                nxt = q.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                            if nxt is None:
                                await q.put(None)
                                break
                            if isinstance(nxt, dict) and nxt.get("event") == "token":
                                merged_text += nxt.get("token", "")
                                merged_count += int(nxt.get("n_tokens", 1))
                            else:
                                leftover = nxt
                                break
                        event = dict(event, token=merged_text, n_tokens=merged_count)
                        yield event
                        if leftover is not None:
                            yield leftover
                            if leftover.get("event") in ("done", "error"):
                                break
                    else:
                        yield event
                        if event.get("event") in ("done", "error"):
                            break"""

assert old_drain in text, "old_drain not found!"
text = text.replace(old_drain, new_drain, 1)

# 2. chat_completions start
old_chat_start = """                accumulated_text = ""
                emitted_tool_call_count = 0
                emitted_role = False
                last_finish_reason = "stop"
                prompt_toks = 0
                compl_toks = 0
                tag_buffer = ""

                # Emit initial assistant role chunk immediately so client receives first event with zero delay
                initial_chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(initial_chunk)}\\n\\n"
                emitted_role = True"""

new_chat_start = """                accumulated_text = ""
                emitted_tool_call_count = 0
                emitted_role = False
                last_finish_reason = "stop"
                prompt_toks = 0
                compl_toks = 0
                tag_buffer = ""
                streamed_any_content = False
                first_token_time = None
                gen_start_time = time.perf_counter()

                def make_chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> Dict[str, Any]:
                    return {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                # Emit initial assistant role chunk immediately so client receives first event with zero delay
                initial_chunk = make_chunk({"role": "assistant"})
                yield f"data: {json.dumps(initial_chunk)}\\n\\n"
                emitted_role = True"""

assert old_chat_start in text, "old_chat_start not found!"
text = text.replace(old_chat_start, new_chat_start, 1)

# 3. token count & timing
old_tok = """                    if event.get("event") == "token":
                        piece = event.get("token", "")
                        accumulated_text += piece"""

new_tok = """                    if event.get("event") == "token":
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        piece = event.get("token", "")
                        compl_toks += int(event.get("n_tokens", 1))
                        accumulated_text += piece"""

assert old_tok in text, "old_tok not found!"
text = text.replace(old_tok, new_tok, 1)

# 4. streamed_any_content flag and make_chunk for content emission
old_emit = """                                    to_send = tag_buffer
                                    tag_buffer = ""
                                    delta = {"content": to_send}
                                    if not emitted_role:
                                        delta["role"] = "assistant"
                                        emitted_role = True
                                    chunk = {
                                        "id": rid,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(chunk)}\\n\\n"
                        else:
                            delta = {"content": piece}
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = {
                                "id": rid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\\n\\n" """

new_emit = """                                    to_send = tag_buffer
                                    tag_buffer = ""
                                    delta = {"content": to_send}
                                    if not emitted_role:
                                        delta["role"] = "assistant"
                                        emitted_role = True
                                    chunk = make_chunk(delta)
                                    yield f"data: {json.dumps(chunk)}\\n\\n"
                                    streamed_any_content = True
                        else:
                            delta = {"content": piece}
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = make_chunk(delta)
                            yield f"data: {json.dumps(chunk)}\\n\\n"
                            streamed_any_content = True"""

# Strip trailing spaces on each line for reliable matching
old_emit_stripped = "\n".join(l.rstrip() for l in old_emit.splitlines())
text_stripped = "\n".join(l.rstrip() for l in text.splitlines())
assert old_emit_stripped in text_stripped, "old_emit not found!"

# Perform replace
idx = text_stripped.find(old_emit_stripped)
before = text_stripped[:idx]
after = text_stripped[idx + len(old_emit_stripped):]
text = before + "\n" + new_emit + "\n" + after

# 5. Smart termination (replace duplicate full text bug with smart tag_buffer / non-streamed flush)
old_tail = """                # Flush tag_buffer if leftover text remained
                if tag_buffer:
                    delta = {"content": tag_buffer}
                    tag_buffer = ""
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\\n\\n"

                # Flush any un-emitted tool calls if present
                if req.tools and "<tool_call>" in accumulated_text:
                    _, final_tool_calls, _ = parse_tool_calls_from_text(accumulated_text, allow_unclosed=True)
                    if final_tool_calls and len(final_tool_calls) > emitted_tool_call_count:
                        for idx in range(emitted_tool_call_count, len(final_tool_calls)):
                            tc = final_tool_calls[idx]
                            delta = {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"],
                                        },
                                    }
                                ]
                            }
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = {
                                "id": rid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\\n\\n"
                        emitted_tool_call_count = len(final_tool_calls)

                # Fallback: if no tool calls were emitted, flush accumulated_text so the client never receives an empty response
                if emitted_tool_call_count == 0 and accumulated_text.strip():
                    delta = {"content": accumulated_text.strip()}
                    if not emitted_role:
                        delta["role"] = "assistant"
                        emitted_role = True
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\\n\\n" """

new_tail = """                # Flush any un-emitted tool calls if present
                if req.tools and "<tool_call>" in accumulated_text:
                    _, final_tool_calls, _ = parse_tool_calls_from_text(accumulated_text, allow_unclosed=True)
                    if final_tool_calls and len(final_tool_calls) > emitted_tool_call_count:
                        for idx in range(emitted_tool_call_count, len(final_tool_calls)):
                            tc = final_tool_calls[idx]
                            delta = {
                                "tool_calls": [
                                    {
                                        "index": idx,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["function"]["name"],
                                            "arguments": tc["function"]["arguments"],
                                        },
                                    }
                                ]
                            }
                            if not emitted_role:
                                delta["role"] = "assistant"
                                emitted_role = True
                            chunk = make_chunk(delta)
                            yield f"data: {json.dumps(chunk)}\\n\\n"
                        emitted_tool_call_count = len(final_tool_calls)

                # Akıllı Sonlandırma: Çift gönderimi önle VE tag_buffer'da kalan '<' kodlarını kurtar
                if emitted_tool_call_count == 0:
                    if not streamed_any_content and accumulated_text.strip():
                        delta = {"content": accumulated_text}
                        if not emitted_role:
                            delta["role"] = "assistant"
                            emitted_role = True
                        yield f"data: {json.dumps(make_chunk(delta))}\\n\\n"
                        streamed_any_content = True
                    elif streamed_any_content and tag_buffer:
                        delta = {"content": tag_buffer}
                        tag_buffer = ""
                        yield f"data: {json.dumps(make_chunk(delta))}\\n\\n" """

old_tail_stripped = "\n".join(l.rstrip() for l in old_tail.splitlines())
text_stripped = "\n".join(l.rstrip() for l in text.splitlines())
assert old_tail_stripped in text_stripped, "old_tail not found!"

idx = text_stripped.find(old_tail_stripped)
before = text_stripped[:idx]
after = text_stripped[idx + len(old_tail_stripped):]
text = before + "\n" + new_tail + "\n" + after

# 6. Stream logging and response headers
old_resp = """                yield f"data: {json.dumps(final_chunk)}\\n\\n"
                yield "data: [DONE]\\n\\n"

            except asyncio.CancelledError:
                logger.info(f"Streaming task cancelled for request {rid}")
                await app.state.backend.cancel(rid)
            finally:
                app.state.active_requests -= 1

        return StreamingResponse(event_generator(), media_type="text/event-stream")"""

new_resp = """                yield f"data: {json.dumps(final_chunk)}\\n\\n"
                yield "data: [DONE]\\n\\n"

                now = time.perf_counter()
                total_time = now - gen_start_time
                ttft_ms = (first_token_time - gen_start_time) * 1000.0 if first_token_time else 0.0
                decode_time = (now - first_token_time) if first_token_time else total_time
                effective_toks = max(1, compl_toks - 1) if first_token_time and compl_toks > 1 else compl_toks
                tps = (effective_toks / decode_time) if decode_time > 0.001 else 0.0
                backend_mtp = getattr(app.state.backend, "mtp", "off")
                logger.info(
                    f"[HTTP-STREAM] id={rid} prompt_tokens={prompt_toks} compl_tokens={compl_toks} "
                    f"ttft={ttft_ms:.1f}ms decode_time={decode_time:.2f}s -> {tps:.2f} TPS (mtp={backend_mtp})"
                )

            except asyncio.CancelledError:
                logger.info(f"Streaming task cancelled for request {rid}")
                await app.state.backend.cancel(rid)
            finally:
                app.state.active_requests -= 1

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )"""

old_resp_stripped = "\n".join(l.rstrip() for l in old_resp.splitlines())
text_stripped = "\n".join(l.rstrip() for l in text.splitlines())
assert old_resp_stripped in text_stripped, "old_resp not found!"

idx = text_stripped.find(old_resp_stripped)
before = text_stripped[:idx]
after = text_stripped[idx + len(old_resp_stripped):]
text = before + "\n" + new_resp + "\n" + after

# 7. Completions streaming headers as well
old_comp_resp = """        return StreamingResponse(stream_generator(), media_type="text/event-stream")"""
new_comp_resp = """        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )"""
assert old_comp_resp in text, "old_comp_resp not found!"
text = text.replace(old_comp_resp, new_comp_resp, 1)

# 8. CLI parser defaults: threads=8, prompt_cache_mb=0, boost_level=2, mtp="pld"
old_parser_thr = 'parser.add_argument("--threads", "-t", type=int, default=16, help="Number of CPU threads (default: 16)")'
new_parser_thr = 'parser.add_argument("--threads", "-t", type=int, default=8, help="Number of CPU threads (default: 8)")'
assert old_parser_thr in text, "old_parser_thr not found!"
text = text.replace(old_parser_thr, new_parser_thr, 1)

old_parser_cache = 'parser.add_argument("--prompt-cache-mb", type=int, default=256, help="Prompt cache memory budget in MB (default: 256)")'
new_parser_cache = 'parser.add_argument("--prompt-cache-mb", type=int, default=0, help="Prompt cache memory budget in MB (default: 0)")'
assert old_parser_cache in text, "old_parser_cache not found!"
text = text.replace(old_parser_cache, new_parser_cache, 1)

old_parser_boost = 'parser.add_argument("--boost-level", type=int, default=1, choices=[1, 2, 3],'
new_parser_boost = 'parser.add_argument("--boost-level", type=int, default=2, choices=[1, 2, 3],'
assert old_parser_boost in text, "old_parser_boost not found!"
text = text.replace(old_parser_boost, new_parser_boost, 1)

old_parser_mtp = 'parser.add_argument("--mtp", choices=["off", "ram", "vram", "atlas", "auto", "hybrid"], default="off",'
new_parser_mtp = 'parser.add_argument("--mtp", choices=["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"], default="pld",'
assert old_parser_mtp in text, "old_parser_mtp not found!"
text = text.replace(old_parser_mtp, new_parser_mtp, 1)

old_parser_mtpmode = 'parser.add_argument("--mtp-mode", choices=["off", "ram", "vram", "atlas", "auto", "hybrid"], default=None,'
new_parser_mtpmode = 'parser.add_argument("--mtp-mode", choices=["off", "ram", "vram", "atlas", "auto", "hybrid", "pld"], default=None,'
assert old_parser_mtpmode in text, "old_parser_mtpmode not found!"
text = text.replace(old_parser_mtpmode, new_parser_mtpmode, 1)

# In main(): mtp default "pld" instead of "off"
old_main_mtp = 'mtp=args.mtp_mode if args.mtp_mode is not None else (args.mtp or "off"),'
new_main_mtp = 'mtp=args.mtp_mode if args.mtp_mode is not None else (args.mtp or "pld"),'
assert old_main_mtp in text, "old_main_mtp not found!"
text = text.replace(old_main_mtp, new_main_mtp, 1)

with open("src/atlas/server.py", "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("SUCCESS: applied all server.py updates.")
