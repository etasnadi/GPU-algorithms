# Lab 04

## Goal

Implement matrix transpose kernel, use GMEM->GMEM directly (`transpose-naive.cu`).

Measure perf with cuda events.

Do perf measurements, compare it with the copy kernel perf from lab 01.

Use `ncu` to create a report to see the reason for the perf differences of the two kernels in `nsight-compute`.

```bash
ncu --set detailed \
  --kernel-name regex:transpose_naive \
  --launch-skip 1 \
  --launch-count 1 \
  --export 04/transpose-naive-profile \
  --force-overwrite \
  ./04/transpose-naive'
```

Set can be `basic`, `detailed`, `full`.