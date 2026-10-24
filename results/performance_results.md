# Performance test results

## Speed, memory and size

| Metric | Value |
|---|---|
| Single prediction, model only (mean / p95) | 22.3 ms / 29.2 ms |
| Single prediction, end to end incl. features (mean / p95) | 34.3 ms / 43.1 ms |
| Batch throughput (11,437 rows) | 43,856 rows/s |
| Peak Python memory during batch prediction | 3.9 MB |
| Classifier file size | 3171 KB |
| RUL regressor file size | 9000 KB |

## Robustness to extra sensor noise

| Extra noise (x sensor sigma) | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 0.637 | 0.848 | 0.727 |
| 0.5 | 0.644 | 0.837 | 0.728 |
| 1 | 0.666 | 0.842 | 0.744 |
| 2 | 0.689 | 0.767 | 0.726 |
| 3 | 0.700 | 0.704 | 0.702 |

## Robustness to missing sensor readings

| Missing fraction | Precision | Recall | F1 |
|---|---|---|---|
| 0% | 0.637 | 0.848 | 0.727 |
| 5% | 0.644 | 0.829 | 0.725 |
| 10% | 0.657 | 0.809 | 0.725 |
| 20% | 0.678 | 0.771 | 0.721 |
| 30% | 0.691 | 0.708 | 0.699 |
