# Atlas Engine V1 Runtime Specification

## Scope
V1 replaces offline trace collection as the central mechanism. Traces remain useful for regression tests, but the runtime makes decisions while inference is running.

## Memory policy

MoE routed expert tensors are host-resident and memory-mapped. Atlas never attempts to allocate the complete 128+ GiB expert corpus in VRAM or RAM.

## Prediction policy

For every layer, a bounded transition table tracks:

`expert_t -> expert_(t+1)`

Only the top `K` candidates are prefetched, with `K <= runtime top-k + 1`. This is deliberately conservative for an 8 GiB VRAM / 16 GiB RAM machine.

## Correctness invariant

Atlas is advisory only. It may prefetch the wrong page, but it cannot alter router logits, selected expert ids, token sampling, or model weights.

## V1 observability

Runtime metrics: router events, prediction hits/misses, prefetch requests, bytes requested, bytes actually touched by the callback, prediction lead time, and session TPS.
