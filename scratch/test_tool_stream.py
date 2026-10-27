import re
import json

def stream_tool(chunks):
    tool_header_buffer = ""
    in_tool_stream = False
    brace_depth = 0
    in_string = False
    escape = False
    started = False
    emitted_chunks = []
    
    for piece in chunks:
        if not in_tool_stream:
            tool_header_buffer += piece
            if "<tool_call>" in tool_header_buffer:
                m_name = re.search(r'["\']name["\']\s*:\s*["\']([a-zA-Z0-9_\.\-]+)["\']', tool_header_buffer)
                m_args = re.search(r'["\']arguments["\']\s*:\s*', tool_header_buffer)
                if m_name and m_args and m_args.start() > m_name.start():
                    in_tool_stream = True
                    args_start = m_args.end()
                    rem = tool_header_buffer[args_start:]
                    tool_header_buffer = ""
                    piece = rem
                else:
                    continue
        if in_tool_stream:
            emit_now = []
            for i, c in enumerate(piece):
                if not started:
                    if c in " \t\r\n":
                        emit_now.append(c)
                        continue
                    elif c == "{":
                        started = True
                        brace_depth = 1
                        emit_now.append(c)
                        continue
                if escape:
                    escape = False
                    emit_now.append(c)
                elif c == "\\" and in_string:
                    escape = True
                    emit_now.append(c)
                elif c == '"':
                    in_string = not in_string
                    emit_now.append(c)
                elif not in_string:
                    if c == "{":
                        brace_depth += 1
                        emit_now.append(c)
                    elif c == "}":
                        brace_depth -= 1
                        emit_now.append(c)
                        if brace_depth == 0:
                            # Finished arguments object!
                            break
                    else:
                        emit_now.append(c)
                else:
                    emit_now.append(c)
            if emit_now:
                emitted_chunks.append("".join(emit_now))
            if started and brace_depth == 0:
                in_tool_stream = False
                break
    return emitted_chunks

# Test case 1: Complex JSON with nested dictionary inside string
chunks1 = [
    '<tool_call>\n',
    '{"name": "edit_file", "arguments": ',
    '{"path": "main.py", ',
    '"content": "def foo():\\n    return {\\"a\\": 1}\\n"',
    '}',
    '}\n</tool_call>'
]
res1 = stream_tool(chunks1)
full1 = "".join(res1)
print("Test 1 Chunks:", len(res1), res1)
print("Test 1 Full:", full1)
parsed1 = json.loads(full1)
assert parsed1["path"] == "main.py"
assert "return {\"a\": 1}" in parsed1["content"]
print("Test 1 PASS!")

# Test case 2: Empty arguments object
chunks2 = ['<tool_call>{"name": "list_files", "arguments": {}}</tool_call>']
res2 = stream_tool(chunks2)
full2 = "".join(res2)
print("Test 2 Chunks:", len(res2), res2)
parsed2 = json.loads(full2)
assert parsed2 == {}
print("Test 2 PASS!")

# Test case 3: Nested JSON objects
chunks3 = [
    '<tool_call>{"name": "test", "arguments": ',
    '{"a": {"b": {"c": 123}}, "d": [1, 2, 3]}',
    '}</tool_call>'
]
res3 = stream_tool(chunks3)
full3 = "".join(res3)
print("Test 3 Chunks:", len(res3), res3)
parsed3 = json.loads(full3)
assert parsed3["a"]["b"]["c"] == 123
print("Test 3 PASS!")
print("ALL STREAMING PARSER TESTS PASS!")
