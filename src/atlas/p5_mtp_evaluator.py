"""P5 MTP Draft Evaluator + API Integration Simulator — Tendou Launcher.

Gerçek inference kanıtı değil; politika ve kabul ölçütü simülatörü.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class DraftQuality:
    draft_length: int
    accepted_tokens: int
    diverged_at: Optional[int]
    is_lossless: bool

class MTPDraftEvaluator:
    def __init__(self, max_draft_length: int = 4, lossless_threshold: float = 1.0):
        self.max_draft_length = max_draft_length
        self.lossless_threshold = lossless_threshold

    def evaluate(self, reference_tokens: List[int], draft_tokens: List[int]) -> DraftQuality:
        diverged_at = None
        accepted_tokens = 0
        min_len = min(len(reference_tokens), len(draft_tokens))
        
        for i in range(min_len):
            if reference_tokens[i] == draft_tokens[i]:
                accepted_tokens += 1
            else:
                diverged_at = i
                break
                
        if diverged_at is None and len(draft_tokens) > len(reference_tokens):
            diverged_at = len(reference_tokens)
            
        is_lossless = (diverged_at is None)
        
        return DraftQuality(
            draft_length=len(draft_tokens),
            accepted_tokens=accepted_tokens,
            diverged_at=diverged_at,
            is_lossless=is_lossless
        )

    def is_safe_default(self, quality: DraftQuality) -> bool:
        return quality.is_lossless

    def get_accept_rate(self, qualities: List[DraftQuality]) -> float:
        if not qualities:
            return 0.0
        total_accepted = sum(q.accepted_tokens for q in qualities)
        total_draft = sum(q.draft_length for q in qualities)
        if total_draft == 0:
            return 0.0
        return total_accepted / total_draft

class APICompatibilityChecker:
    def __init__(self):
        pass

    def check_cancel_safe(self, in_flight_drafts: int) -> bool:
        return in_flight_drafts == 0

    def check_cache_behavior(self, prompt_tokens_cached: int, total_prompt_tokens: int) -> dict:
        fresh_tokens = total_prompt_tokens - prompt_tokens_cached
        cache_hit_rate = prompt_tokens_cached / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
        return {
            "cache_hit_rate": cache_hit_rate,
            "cache_tokens": prompt_tokens_cached,
            "fresh_tokens": fresh_tokens
        }

    def check_opencode_stream_compatible(self, stream_events: List[str]) -> bool:
        if not stream_events or len(stream_events) < 2:
            return False
        if stream_events[0] != "start" or stream_events[-1] != "done":
            return False
        for event in stream_events[1:-1]:
            if event in ("start", "done"):
                return False
        return True

class MTPSafetyGate:
    def __init__(self, evaluator: MTPDraftEvaluator, checker: APICompatibilityChecker):
        self.evaluator = evaluator
        self.checker = checker

    def approve_for_default(self, qualities: List[DraftQuality]) -> Tuple[bool, str]:
        if not qualities:
            return False, "divergence_detected"
        
        for q in qualities:
            if not q.is_lossless:
                return False, "divergence_detected"
                
        accept_rate = self.evaluator.get_accept_rate(qualities)
        if accept_rate < 0.8:
            return False, "low_accept_rate"
            
        return True, "approved"
