# GEMM Benchmark

## Environment

- Build type: Release
- Compiler: GCC ...
- CPU: ...
- Date: 2026-08-03
- Git commit: ...

## Configurations

- Matrix sizes: 128, 256, 512, 1024, 2048
- Block sizes: 16, 32, 64, 128
- Warmup count: ...
- Iteration count: ...
- Data type: float32
- Thread count: single-threaded

## Findings

1. IKJ is consistently faster than IJK.
2. Blocked GEMM is approximately 20% faster than IKJ for most tested sizes.
3. IJK performance drops sharply at 2048³.
4. Block size 32 is a reasonable default, although block size 128 performs
   best for some larger square matrices.