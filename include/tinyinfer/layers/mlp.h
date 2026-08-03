#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/linear.h"
#include "tinyinfer/ops/swiglu.h"
#include "tinyinfer/ops/gemm.h"

namespace tinyinfer
{
    class MLP
    {
    public:
        MLP(
            Matrix gate_weight,
            Matrix gate_bias,
            Matrix up_weight,
            Matrix up_bias,
            Matrix down_weight,
            Matrix down_bias,
            const GemmBackend &gemm_backend = tinyinfer::GemmIKJBackend());

        Matrix forward(const Matrix &input) const;

    private:
        Matrix gate_weight_;
        Matrix gate_bias_;
        Matrix up_weight_;
        Matrix up_bias_;
        Matrix down_weight_;
        Matrix down_bias_;
        const GemmBackend& gemm_backend_;
    };
}