#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "tinyinfer/layers/mlp.h"
#include "tinyinfer/ops/linear.h"
#include "tinyinfer/ops/swiglu.h"

namespace tinyinfer
{
    MLP::MLP(
        Matrix gate_weight,
        Matrix gate_bias,
        Matrix up_weight,
        Matrix up_bias,
        Matrix down_weight,
        Matrix down_bias) : gate_weight_(gate_weight),
                            gate_bias_(gate_bias),
                            up_weight_(up_weight),
                            up_bias_(up_bias),
                            down_weight_(down_weight),
                            down_bias_(down_bias)
    {
        // Gate weight 尺寸
        std::size_t gate_weight_rows = gate_weight.rows();
        std::size_t gate_weight_cols = gate_weight.cols();

        // Gate bias 尺寸（一维向量，行数=元素个数，列数=1 或 0）
        std::size_t gate_bias_rows = gate_bias.rows();
        std::size_t gate_bias_cols = gate_bias.cols();

        // Up weight 尺寸
        std::size_t up_weight_rows = up_weight.rows();
        std::size_t up_weight_cols = up_weight.cols();

        // Up bias 尺寸
        std::size_t up_bias_rows = up_bias.rows();
        std::size_t up_bias_cols = up_bias.cols();

        // Down weight 尺寸
        std::size_t down_weight_rows = down_weight.rows();
        std::size_t down_weight_cols = down_weight.cols();

        // Down bias 尺寸
        std::size_t down_bias_rows = down_bias.rows();
        std::size_t down_bias_cols = down_bias.cols();

        if (gate_weight_rows == 0 || gate_weight_cols == 0 ||
            up_weight_rows == 0 || up_weight_cols == 0 ||
            down_weight_rows == 0 || down_weight_cols == 0 ||
            gate_bias_rows == 0 || gate_bias_cols == 0 ||
            up_bias_rows == 0 || up_bias_cols == 0 ||
            down_bias_rows == 0 || down_bias_cols == 0)
        {
            throw std::invalid_argument("MLP parameters must not be empty.");
        }

        if (gate_weight_rows != up_weight_rows)
        {
            throw std::invalid_argument("MLP gate and up weights must have the same input size.");
        }

        if (gate_weight_cols != up_weight_cols)
        {
            throw std::invalid_argument("MLP gate and up weights must have the same intermediate size.");
        }

        if (gate_bias_rows != 1 || gate_bias_cols != gate_weight_cols)
        {
            throw std::invalid_argument("MLP gate bias shape mismatch.");
        }

        if (up_bias_rows != 1 || up_bias_cols != up_weight_cols)
        {
            throw std::invalid_argument("MLP up bias shape mismatch.");
        }

        if (down_weight_rows != gate_weight_cols)
        {
            throw std::invalid_argument("MLP down weight input size must match intermediate size.");
        }

        if (down_weight_cols != gate_weight_rows)
        {
            throw std::invalid_argument("MLP down weight output size must match hidden size.");
        }

        if (down_bias_rows != 1 || down_bias_cols != down_weight_cols)
        {
            throw std::invalid_argument("MLP down bias shape mismatch.");
        }
    }

    Matrix MLP::forward(const Matrix &input) const
    {
        std::size_t input_rows = input.rows();
        std::size_t input_cols = input.cols();

        if (input.rows() == 0 || input.cols() == 0)
        {
            throw std::invalid_argument("MLP input must not be empty.");
        }

        if (input_cols != gate_weight_.rows())
        {
            throw std::invalid_argument("input Matrix cols must be same with gate weight rows.");
        }

        Matrix gate = Linear(input, gate_weight_, gate_bias_);

        Matrix up = Linear(input, up_weight_, up_bias_);

        Matrix hidden = SwiGLu(gate, up);

        Matrix output = Linear(hidden, down_weight_, down_bias_);

        return output;
    }

}