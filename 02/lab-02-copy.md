# Lab 01

## Goal

Implement the copy kernel `copy-kernel.cu` that copies from device buffer A to device buffer B.

Intentionally overindex the inpur buffer in `copy-kernel-error.cu`, then check the problems with `compute-sanitizer`. We should see an error.

Add proper error handling to `copy-kernel`.

Add wall-clock time measurement with cuda events.