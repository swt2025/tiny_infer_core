# 05 Softmax

## 实验目标

实现 Transformer 推理中常用的按行 softmax，并理解数值稳定 softmax 的必要性。

## 当前实现

实现了两个版本：

- SoftmaxNaive
- SoftmaxStable

Softmax 按行计算：

```text
output[i][j] = exp(input[i][j]) / sum_j exp(input[i][j])
