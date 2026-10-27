# Atlas simulator: interpretation rules

The simulator is a controlled hypothesis test. It does not prove that a real
GPU runtime will reach the reported TPS.

## Ground truth

The trace's expert selections are immutable. Atlas cannot replace the router.
A prefetch hit means the bytes arrived early enough; a prediction error only
causes wasted I/O/cache pollution.

## Main decision metric

Do not optimize hit-rate alone. Compare:

- exposed I/O milliseconds/token
- P95/P99 exposed I/O
- NVMe MB/token
- RAM->VRAM MB/token
- prefetch precision/recall
- effective TPS under a stated compute window

## Required next experiment

The structural GGUF map must be joined to the trace-derived logical map. Only
then can we test physical chunking questions such as:

- whole expert vs tensor-group prefetch
- contiguous region vs scattered tensor reads
- quant-block aligned reads
- RAM persistence across requests
- multi-layer prefetch horizon
- confidence thresholds

No semantic labels such as "reasoning expert" or "coding expert" should be
introduced unless an independent experiment demonstrates that signal.

## v3.1 decision rule

The 35B target result makes `causal_region` the leading hypothesis, but not yet
a runtime design. Its large prediction set can reduce exposed mean I/O while
increasing PCIe traffic and VRAM churn. Region size is therefore swept as a
sensitivity parameter. The simulator reports mean and P95 winners separately;
there is intentionally no arbitrary weighted score.

Next physical validation requires real GGUF tensor offsets. Logical regions
must not be treated as physically contiguous until the GGUF structural map
confirms it.
