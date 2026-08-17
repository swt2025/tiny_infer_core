#include "tinyinfer/core/matrix.h"

// attention流程：
// 1. 输入：一个(m,n)的矩阵表示m个n维的token
// 2. 输出：注意力结果
namespace tinyinfer
{
    class Attention
    {
    public:
        Attention(
            const Matrix &q_weight, const Matrix &q_bias,
            const Matrix &k_weight, const Matrix &k_bias,
            const Matrix &v_weight, const Matrix &v_bias,
            const Matrix &o_weight, const Matrix &o_bias,
            const GemmBackend &gemm_backend,
            const float RoPE_base = 10000.0f);

        Matrix forward(const Matrix &input, const Matrix &abs_position) const;

    private:
        Matrix q_weight_;
        Matrix q_bias_;
        Matrix k_weight_;
        Matrix k_bias_;
        Matrix v_weight_;
        Matrix v_bias_;

        Matrix o_weight_;
        Matrix o_bias_;

        const GemmBackend &gemm_backend_;

        float RoPE_base_;
    };
}