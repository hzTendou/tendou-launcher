"""
Tendou Studio — Persistent Chat History & Context-Growth Telemetry Store.

Stores ChatGPT-style multi-turn conversations as atomic per-chat JSON files under
`data/chats/` (gitignored), tracking per-turn TTFT, decode TPS, prefill TPS,
prompt/completion token counts, and cumulative context growth so users can inspect
real-world long-context performance across sessions and tab switches.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHATS_DIR = REPO_ROOT / "data" / "chats"

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")

DEFAULT_SYSTEM_PROMPT = (
    "You are Tendou Assistant, a concise, accurate software and systems engineering collaborator."
)


class ChatTurnMetrics(BaseModel):
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    prefill_tps: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    context_limit: int = 65536
    prefix_fork_kind: Optional[str] = None
    finish_reason: Optional[str] = None


class ChatMessageRecord(BaseModel):
    id: str = Field(..., min_length=1, max_length=80)
    role: Literal["user", "assistant"]
    content: str = ""
    thinking: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    model_id: Optional[str] = None
    metrics: Optional[ChatTurnMetrics] = None
    stopped: bool = False


class ChatParams(BaseModel):
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=0, le=262144)
    stream: bool = True


class ChatConversation(BaseModel):
    id: str
    title: str = "New Chat"
    model_id: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    pinned: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    params: ChatParams = Field(default_factory=ChatParams)
    messages: List[ChatMessageRecord] = Field(default_factory=list)


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    params: Optional[ChatParams] = None


class ChatUpdateRequest(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    system_prompt: Optional[str] = None
    pinned: Optional[bool] = None
    params: Optional[ChatParams] = None
    messages: Optional[List[ChatMessageRecord]] = None


def derive_chat_title(text: str, max_len: int = 56) -> str:
    """Derives a clean, human-readable title from the first non-empty line of the user prompt."""
    raw = (text or "").strip()
    if not raw:
        return "New Chat"
    first_nonempty = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    first_line = re.sub(r"\s+", " ", first_nonempty).strip(" #*`\"'")
    if not first_line:
        return "New Chat"
    if len(first_line) <= max_len:
        return first_line
    cut = first_line[:max_len].rsplit(" ", 1)[0].strip()
    return (cut or first_line[:max_len]).rstrip(".,;:") + "…"


def build_context_curve(conv: ChatConversation) -> List[Dict[str, Any]]:
    """Extracts ordered turn-by-turn context growth vs TPS/TTFT points."""
    points: List[Dict[str, Any]] = []
    turn_idx = 0
    for msg in conv.messages:
        if msg.role != "assistant" or msg.metrics is None:
            continue
        turn_idx += 1
        m = msg.metrics
        points.append({
            "turn": turn_idx,
            "message_id": msg.id,
            "prompt_tokens": m.prompt_tokens,
            "completion_tokens": m.completion_tokens,
            "total_tokens": m.total_tokens or (m.prompt_tokens + m.completion_tokens),
            "ttft_ms": round(m.ttft_ms, 2),
            "decode_tps": round(m.decode_tps, 2),
            "prefill_tps": round(m.prefill_tps, 2) if m.prefill_tps is not None else None,
            "duration_ms": round(m.duration_ms, 1),
            "context_limit": m.context_limit,
            "prefix_fork_kind": m.prefix_fork_kind,
        })
    return points


def summarize_conversation(conv: ChatConversation) -> Dict[str, Any]:
    """Produces a compact sidebar summary with latest context & speed stats."""
    last_preview = ""
    total_tokens = 0
    last_decode_tps = 0.0
    last_ttft_ms = 0.0
    turns = 0

    for msg in conv.messages:
        if msg.role == "assistant":
            turns += 1
            if msg.metrics:
                total_tokens = max(
                    total_tokens,
                    msg.metrics.total_tokens or (msg.metrics.prompt_tokens + msg.metrics.completion_tokens),
                )
                if msg.metrics.decode_tps > 0:
                    last_decode_tps = round(msg.metrics.decode_tps, 1)
                if msg.metrics.ttft_ms > 0:
                    last_ttft_ms = round(msg.metrics.ttft_ms, 1)
        if msg.content.strip():
            last_preview = re.sub(r"\s+", " ", msg.content.strip())[:110]

    return {
        "id": conv.id,
        "title": conv.title,
        "model_id": conv.model_id,
        "pinned": conv.pinned,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "message_count": len(conv.messages),
        "turn_count": turns,
        "total_tokens": total_tokens,
        "last_decode_tps": last_decode_tps,
        "last_ttft_ms": last_ttft_ms,
        "last_preview": last_preview,
    }


def export_conversation_markdown(conv: ChatConversation) -> str:
    lines: List[str] = [
        f"# {conv.title}",
        "",
        f"- **Model**: `{conv.model_id or 'local'}`",
        f"- **Messages**: {len(conv.messages)}",
        "",
    ]
    if conv.system_prompt.strip():
        quoted_sys = [f"> {ln}" for ln in conv.system_prompt.strip().splitlines()]
        lines.extend(["## System Prompt", "", *quoted_sys, ""])
    for msg in conv.messages:
        role_hdr = "User" if msg.role == "user" else f"Assistant ({msg.model_id or conv.model_id or 'Tendou'})"
        lines.extend([f"### {role_hdr}", ""])
        if msg.thinking and msg.thinking.strip():
            lines.extend(["<details><summary>Reasoning</summary>", "", msg.thinking.strip(), "", "</details>", ""])
        lines.append(msg.content)
        if msg.metrics:
            m = msg.metrics
            lines.extend([
                "",
                f"*TTFT {m.ttft_ms:.0f} ms · Decode {m.decode_tps:.1f} tok/s · Context {m.total_tokens} tok ({m.prompt_tokens} in / {m.completion_tokens} out)*",
            ])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


class ChatStore:
    """Thread-safe atomic JSON directory store for Tendou Studio chats."""

    def __init__(self, chats_dir: Path = DEFAULT_CHATS_DIR):
        self.chats_dir = Path(chats_dir)
        self._lock = threading.RLock()

    def _validate_id(self, chat_id: str) -> str:
        if not chat_id or not _SAFE_ID_RE.match(chat_id):
            raise ValueError(f"Invalid chat_id: {chat_id!r}")
        return chat_id

    def _path_for(self, chat_id: str) -> Path:
        safe_id = self._validate_id(chat_id)
        target = (self.chats_dir / f"{safe_id}.json").resolve()
        base = self.chats_dir.resolve()
        if not target.is_relative_to(base):
            raise ValueError("Path traversal blocked")
        return target

    def _write_atomic(self, conv: ChatConversation) -> None:
        self.chats_dir.mkdir(parents=True, exist_ok=True)
        path = self._path_for(conv.id)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
        try:
            tmp.write_text(json.dumps(conv.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def _read_file(self, path: Path) -> Optional[ChatConversation]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return ChatConversation(**raw)
        except Exception:
            return None
        return None

    def list_chats(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        q = (query or "").strip().lower()
        with self._lock:
            if not self.chats_dir.is_dir():
                return []
            convs: List[ChatConversation] = []
            for p in self.chats_dir.glob("*.json"):
                conv = self._read_file(p)
                if conv is None:
                    continue
                if q:
                    in_title = q in conv.title.lower()
                    in_msgs = any(q in (m.content or "").lower() for m in conv.messages)
                    if not (in_title or in_msgs):
                        continue
                convs.append(conv)
            convs.sort(key=lambda c: (not c.pinned, -c.updated_at))
            return [summarize_conversation(c) for c in convs]

    def get_chat(self, chat_id: str) -> Optional[ChatConversation]:
        with self._lock:
            path = self._path_for(chat_id)
            if not path.is_file():
                return None
            return self._read_file(path)

    def create_chat(self, req: Optional[ChatCreateRequest] = None) -> ChatConversation:
        now = time.time()
        chat_id = f"chat_{int(now * 1000)}_{uuid.uuid4().hex[:8]}"
        raw_title = (req.title.strip() if req and req.title and req.title.strip() else "New Chat")[:160]
        conv = ChatConversation(
            id=chat_id,
            title=raw_title,
            model_id=req.model_id if req else None,
            system_prompt=(
                req.system_prompt
                if (req is not None and req.system_prompt is not None)
                else DEFAULT_SYSTEM_PROMPT
            ),
            created_at=now,
            updated_at=now,
            params=req.params if (req and req.params is not None) else ChatParams(),
            messages=[],
        )
        with self._lock:
            self._write_atomic(conv)
        return conv

    def update_chat(self, chat_id: str, req: ChatUpdateRequest) -> Optional[ChatConversation]:
        with self._lock:
            conv = self.get_chat(chat_id)
            if conv is None:
                return None
            if req.title is not None:
                conv.title = (req.title.strip() or "New Chat")[:160]
            if req.model_id is not None:
                conv.model_id = req.model_id
            if req.system_prompt is not None:
                conv.system_prompt = req.system_prompt
            if req.pinned is not None:
                conv.pinned = bool(req.pinned)
            if req.params is not None:
                conv.params = req.params
            if req.messages is not None:
                conv.messages = list(req.messages)
                if conv.title == "New Chat":
                    first_user = next((m for m in conv.messages if m.role == "user" and m.content.strip()), None)
                    if first_user:
                        conv.title = derive_chat_title(first_user.content)
            conv.updated_at = time.time()
            self._write_atomic(conv)
            return conv

    def upsert_message(self, chat_id: str, message: ChatMessageRecord) -> Optional[ChatConversation]:
        with self._lock:
            conv = self.get_chat(chat_id)
            if conv is None:
                return None
            replaced = False
            for idx, existing in enumerate(conv.messages):
                if existing.id == message.id:
                    conv.messages[idx] = message
                    replaced = True
                    break
            if not replaced:
                conv.messages.append(message)
            if conv.title == "New Chat" and message.role == "user" and message.content.strip():
                conv.title = derive_chat_title(message.content)
            if message.model_id:
                conv.model_id = message.model_id
            conv.updated_at = time.time()
            self._write_atomic(conv)
            return conv

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            path = self._path_for(chat_id)
            if not path.is_file():
                return False
            path.unlink()
            return True
