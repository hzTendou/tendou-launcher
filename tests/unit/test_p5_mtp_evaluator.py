import pytest
from src.atlas.p5_mtp_evaluator import (
    DraftQuality,
    MTPDraftEvaluator,
    APICompatibilityChecker,
    MTPSafetyGate
)

def test_draft_quality_lossless_true():
    q = DraftQuality(draft_length=4, accepted_tokens=4, diverged_at=None, is_lossless=True)
    assert q.is_lossless is True

def test_draft_quality_lossless_false():
    q = DraftQuality(draft_length=4, accepted_tokens=2, diverged_at=2, is_lossless=False)
    assert q.is_lossless is False

def test_evaluator_exact_match():
    evaluator = MTPDraftEvaluator()
    ref = [10, 20, 30, 40]
    draft = [10, 20, 30, 40]
    q = evaluator.evaluate(ref, draft)
    assert q.draft_length == 4
    assert q.accepted_tokens == 4
    assert q.diverged_at is None
    assert q.is_lossless is True

def test_evaluator_diverge_first_token():
    evaluator = MTPDraftEvaluator()
    ref = [15, 20, 30, 40]
    draft = [10, 20, 30, 40]
    q = evaluator.evaluate(ref, draft)
    assert q.draft_length == 4
    assert q.accepted_tokens == 0
    assert q.diverged_at == 0
    assert q.is_lossless is False

def test_evaluator_diverge_middle():
    evaluator = MTPDraftEvaluator()
    ref = [10, 20, 35, 40]
    draft = [10, 20, 30, 40]
    q = evaluator.evaluate(ref, draft)
    assert q.draft_length == 4
    assert q.accepted_tokens == 2
    assert q.diverged_at == 2
    assert q.is_lossless is False

def test_evaluator_empty_draft():
    evaluator = MTPDraftEvaluator()
    ref = [10, 20]
    draft = []
    q = evaluator.evaluate(ref, draft)
    assert q.draft_length == 0
    assert q.accepted_tokens == 0
    assert q.diverged_at is None
    assert q.is_lossless is True

def test_evaluator_draft_longer_than_ref():
    evaluator = MTPDraftEvaluator()
    ref = [10, 20]
    draft = [10, 20, 30]
    q = evaluator.evaluate(ref, draft)
    assert q.draft_length == 3
    assert q.accepted_tokens == 2
    assert q.diverged_at == 2
    assert q.is_lossless is False

def test_evaluator_is_safe_default():
    evaluator = MTPDraftEvaluator()
    q1 = DraftQuality(draft_length=2, accepted_tokens=2, diverged_at=None, is_lossless=True)
    q2 = DraftQuality(draft_length=2, accepted_tokens=1, diverged_at=1, is_lossless=False)
    assert evaluator.is_safe_default(q1) is True
    assert evaluator.is_safe_default(q2) is False

def test_evaluator_get_accept_rate_empty():
    evaluator = MTPDraftEvaluator()
    assert evaluator.get_accept_rate([]) == 0.0

def test_evaluator_get_accept_rate_normal():
    evaluator = MTPDraftEvaluator()
    qualities = [
        DraftQuality(draft_length=4, accepted_tokens=4, diverged_at=None, is_lossless=True),
        DraftQuality(draft_length=4, accepted_tokens=2, diverged_at=2, is_lossless=False)
    ]
    assert evaluator.get_accept_rate(qualities) == 6 / 8

def test_api_checker_cancel_safe():
    checker = APICompatibilityChecker()
    assert checker.check_cancel_safe(in_flight_drafts=0) is True

def test_api_checker_cancel_unsafe():
    checker = APICompatibilityChecker()
    assert checker.check_cancel_safe(in_flight_drafts=2) is False

def test_api_checker_cache_behavior():
    checker = APICompatibilityChecker()
    res = checker.check_cache_behavior(prompt_tokens_cached=100, total_prompt_tokens=200)
    assert res["cache_hit_rate"] == 0.5
    assert res["cache_tokens"] == 100
    assert res["fresh_tokens"] == 100

def test_api_checker_stream_compatible():
    checker = APICompatibilityChecker()
    events = ["start", "chunk", "chunk", "done"]
    assert checker.check_opencode_stream_compatible(events) is True

def test_api_checker_stream_incompatible():
    checker = APICompatibilityChecker()
    events = ["chunk", "done"]
    assert checker.check_opencode_stream_compatible(events) is False
    events2 = ["start", "chunk"]
    assert checker.check_opencode_stream_compatible(events2) is False
    events3 = []
    assert checker.check_opencode_stream_compatible(events3) is False
    events4 = ["start", "start", "done"]
    assert checker.check_opencode_stream_compatible(events4) is False
    events5 = ["start", "done", "done"]
    assert checker.check_opencode_stream_compatible(events5) is False


def test_safety_gate_approve_all_lossless():
    evaluator = MTPDraftEvaluator()
    checker = APICompatibilityChecker()
    gate = MTPSafetyGate(evaluator, checker)
    
    qualities = [
        DraftQuality(draft_length=4, accepted_tokens=4, diverged_at=None, is_lossless=True),
        DraftQuality(draft_length=4, accepted_tokens=4, diverged_at=None, is_lossless=True)
    ]
    approved, reason = gate.approve_for_default(qualities)
    assert approved is True
    assert reason == "approved"

def test_safety_gate_reject_divergence():
    evaluator = MTPDraftEvaluator()
    checker = APICompatibilityChecker()
    gate = MTPSafetyGate(evaluator, checker)
    
    qualities = [
        DraftQuality(draft_length=4, accepted_tokens=4, diverged_at=None, is_lossless=True),
        DraftQuality(draft_length=4, accepted_tokens=2, diverged_at=2, is_lossless=False)
    ]
    approved, reason = gate.approve_for_default(qualities)
    assert approved is False
    assert reason == "divergence_detected"

def test_safety_gate_reject_low_accept_rate():
    # If accept_rate < 0.8, it rejects. To test this without triggering divergence,
    # we would need qualities that are lossless but somehow have low accept rate?
    # Wait, if draft_length is 4, but accepted_tokens is 2, and diverged_at is None,
    # that would mean is_lossless=True, which can happen if reference is shorter than draft,
    # but wait, if reference is shorter, diverged_at is set to len(reference) -> is_lossless=False.
    # What if draft is shorter than max draft length but exact match?
    # Empty draft: draft_length=0, accepted=0.
    evaluator = MTPDraftEvaluator()
    checker = APICompatibilityChecker()
    gate = MTPSafetyGate(evaluator, checker)
    
    # Let's bypass the is_lossless check but fail the accept rate.
    # Wait, accept_rate is total_accepted / total_draft.
    # If we have draft_length=4, accepted=4 (rate 1.0)
    # Can we have draft_length=4, accepted=2, and is_lossless=True?
    # Only if we hack the DraftQuality object manually.
    q = DraftQuality(draft_length=10, accepted_tokens=5, diverged_at=None, is_lossless=True)
    qualities = [q]
    
    approved, reason = gate.approve_for_default(qualities)
    assert approved is False
    assert reason == "low_accept_rate"
