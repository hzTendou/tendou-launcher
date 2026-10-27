import pytest
from src.atlas.p3_async_transfer import PinnedStagingBuffer, H2DTransferQueue, CUDAEventFence, AsyncTransferPipeline

def test_pinned_buffer_acquire_release():
    buf = PinnedStagingBuffer(2, 1024)
    s1 = buf.acquire_slot()
    s2 = buf.acquire_slot()
    assert s1 is not None and s2 is not None
    assert buf.acquire_slot() is None
    buf.release_slot(s1)
    assert buf.acquire_slot() is not None

def test_pinned_buffer_is_full():
    buf = PinnedStagingBuffer(1, 1024)
    assert not buf.is_full()
    buf.acquire_slot()
    assert buf.is_full()

def test_pinned_buffer_capacity():
    buf = PinnedStagingBuffer(3, 100)
    slots = []
    for _ in range(3):
        slots.append(buf.acquire_slot())
    assert all(s is not None for s in slots)
    assert buf.acquire_slot() is None

def test_h2d_enqueue_dequeue():
    q = H2DTransferQueue(2)
    assert q.enqueue(0, 1, 0)
    assert q.enqueue(0, 2, 1)
    assert not q.enqueue(0, 3, 2)
    assert q.pending_count == 2
    
    item = q.dequeue()
    assert item == (0, 1, 0)
    assert q.pending_count == 1
    assert q.completed_count == 1

def test_h2d_queue_depth():
    q = H2DTransferQueue(1)
    assert q.enqueue(1, 1, 0)
    assert not q.enqueue(1, 2, 1)

def test_h2d_dequeue_empty():
    q = H2DTransferQueue(1)
    assert q.dequeue() is None

def test_cuda_event_fence_order():
    fence = CUDAEventFence()
    key = (0, 0)
    fence.record_transfer_start(key)
    assert not fence.transfer_complete_before_compute(key)
    fence.record_transfer_complete(key)
    assert fence.transfer_complete_before_compute(key)
    fence.record_compute_start(key)
    
def test_cuda_event_fence_wrong_order():
    fence = CUDAEventFence()
    key = (1, 1)
    fence.record_compute_start(key)
    assert not fence.transfer_complete_before_compute(key)
    
def test_async_pipeline_submit():
    pipe = AsyncTransferPipeline(2, 100, 2)
    assert pipe.submit_expert(0, 1, 50)
    assert pipe.submit_expert(0, 2, 50)
    assert not pipe.submit_expert(0, 3, 50) # buffer full

def test_async_pipeline_size_limit():
    pipe = AsyncTransferPipeline(2, 100, 2)
    assert not pipe.submit_expert(0, 1, 150)

def test_async_pipeline_drain():
    pipe = AsyncTransferPipeline(2, 100, 2)
    pipe.submit_expert(0, 1, 50)
    completed = pipe.drain_completed()
    assert completed == [(0, 1)]
    assert not pipe.staging_buffer.is_full()

def test_async_pipeline_lossless_fallback():
    pipe = AsyncTransferPipeline(1, 100, 1)
    assert pipe.submit_expert(0, 1, 50)
    # queue/buffer is full, next expert fails (CPU fallback)
    assert not pipe.submit_expert(0, 2, 50)
    
def test_async_pipeline_queue_full():
    pipe = AsyncTransferPipeline(5, 100, 1)
    assert pipe.submit_expert(0, 1, 50)
    assert not pipe.submit_expert(0, 2, 50) # queue full
    
def test_async_pipeline_lifecycle():
    pipe = AsyncTransferPipeline(1, 100, 1)
    pipe.submit_expert(0, 1, 50)
    key = (0, 1)
    assert not pipe.event_fence.transfer_complete_before_compute(key)
    pipe.drain_completed()
    assert pipe.event_fence.transfer_complete_before_compute(key)

def test_async_pipeline_cancel_handling():
    # Model cancellation by dropping an un-completed transfer
    pipe = AsyncTransferPipeline(2, 100, 2)
    assert pipe.submit_expert(0, 1, 50)
    assert pipe.submit_expert(0, 2, 50)
    
    # Cancel the first one
    assert pipe.cancel_transfer(0, 1)
    
    # Buffer should now have space
    assert pipe.submit_expert(0, 3, 50)
    
    # Remaining should be 2 and 3
    completed = pipe.drain_completed()
    assert set(completed) == {(0, 2), (0, 3)}

def test_async_pipeline_duplicate_submit_rejected():
    pipe = AsyncTransferPipeline(2, 100, 2)
    assert pipe.submit_expert(0, 1, 50)
    # Duplicate submit must be rejected and not leak staging slots
    assert not pipe.submit_expert(0, 1, 50)
    assert len(pipe.staging_buffer.used_slots) == 1
    assert len(pipe.staging_buffer.available_slots) == 1
    # Can still submit another distinct expert
    assert pipe.submit_expert(0, 2, 50)
    assert len(pipe.staging_buffer.used_slots) == 2

def test_cuda_event_fence_premature_compute_hazard():
    fence = CUDAEventFence()
    key = (2, 5)
    fence.record_transfer_start(key)
    # Premature compute before transfer complete
    fence.record_compute_start(key)
    # Transfer finishes later
    fence.record_transfer_complete(key)
    # Hazard must be detected; transfer was NOT complete before compute
    assert not fence.transfer_complete_before_compute(key)
    assert fence.has_hazard(key)

def test_async_pipeline_clear():
    pipe = AsyncTransferPipeline(2, 100, 2)
    pipe.submit_expert(0, 1, 50)
    pipe.submit_expert(0, 2, 50)
    assert len(pipe.staging_buffer.used_slots) == 2
    pipe.clear()
    assert len(pipe.staging_buffer.used_slots) == 0
    assert len(pipe.staging_buffer.available_slots) == 2
    assert pipe.transfer_queue.is_empty()

def test_cuda_event_fence_untransferred_no_hazard():
    fence = CUDAEventFence()
    key = (1, 10)
    # Untransferred expert computes (e.g. CPU execution)
    fence.record_compute_start(key)
    assert not fence.has_hazard(key)
    assert not fence.transfer_complete_before_compute(key)

def test_cuda_event_fence_cancellation_no_hazard():
    fence = CUDAEventFence()
    key = (3, 7)
    fence.record_transfer_start(key)
    fence.record_transfer_cancelled(key)
    fence.record_compute_start(key)
    assert not fence.has_hazard(key)

def test_async_pipeline_invalid_bounds():
    pipe = AsyncTransferPipeline(2, 100, 2)
    assert not pipe.submit_expert(-1, 0, 50)
    assert not pipe.submit_expert(48, 0, 50)
    assert not pipe.submit_expert(0, -1, 50)
    assert not pipe.submit_expert(0, 512, 50)
    assert len(pipe.staging_buffer.used_slots) == 0

def test_async_pipeline_cancel_clears_fence():
    pipe = AsyncTransferPipeline(2, 100, 2)
    pipe.submit_expert(0, 1, 50)
    assert pipe.cancel_transfer(0, 1)
    pipe.event_fence.record_compute_start((0, 1))
    assert not pipe.event_fence.has_hazard((0, 1))

