# 06 RMSNorm

## 实验目标

实现 Transformer Decoder 中常用的 RMSNorm 算子。

## 输入输出

当前约定：

- input: `[seq_len, hidden_size]`
- weight: `[1, hidden_size]`
- output: `[seq_len, hidden_size]`

RMSNorm 按行计算，每一行对应一个 token 的 hidden state。

## 公式

```text
mean_square = sum(x_j * x_j) / hidden_size
inv_rms = 1 / sqrt(mean_square + eps)
y_j = x_j * inv_rms * weight_j
