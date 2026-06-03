# 01 GEMM Naive

## 今日完成内容

- 实现 GemmNaive
- 支持 Matrix A 和 Matrix B 的矩阵乘法
- 添加 shape mismatch 检查
- 添加固定值测试、单位矩阵测试、零矩阵测试、shape mismatch 测试

## 当前实现

当前 GEMM 使用最朴素的三重循环：

```cpp
for i
  for j
    for k
      c(i, j) += a(i, k) * b(k, j)
