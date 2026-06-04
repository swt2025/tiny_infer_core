# 03 GEMM Loop Order

## 实验目标

对比 `ijk` 和 `ikj` 两种 GEMM 循环顺序在 row-major Matrix 下的性能差异。

## 实现版本

### GemmIJK

```cpp
for i
  for j
    for k
      c(i, j) += a(i, k) * b(k, j)
