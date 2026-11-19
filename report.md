# Routing decision model — key metrics

**Strands Decider 2B** · checkpoint `run13` · measured on a fresh 80-case holdout

| Metric | Result |
|---|---|
| **Overall accuracy** | **90.0%** |
| high tier → local | 96.0% |
| some tier → private | 83.3% |
| none tier → public | 92.0% |
| Sensitive content reaching a public API | **0** |
| Calibration (guard ECE) | 0.143 |
| Latency p50 / p95 | 115.8 / 126.6 ms |
| Thresholds that must be fitted | 1 |
