#include <cmath>
#include <stdexcept>
#include <cstddef>
#include "tinyinfer/ops/swiglu.h"

namespace tinyinfer
{
    Matrix SwiGLu(const Matrix &gate, const Matrix &up)
    {
        std::size_t gate_rows = gate.rows();
        std::size_t gate_cols = gate.cols();
        std::size_t up_rows = up.rows();
        std::size_t up_cols = up.cols();

        if (gate_rows == 0 || gate_cols == 0)
        {
            throw std::invalid_argument("SwiGLU input matrix must not be empty.");
        }

        if (gate_rows != up_rows || gate_cols != up_cols)
        {
            throw std::invalid_argument("SwiGLU input matrices must have the same shape.");
        }

        Matrix output(gate_rows, gate_cols, 0.0f);

        Matrix gate_silu = SiLU(gate);
        for (std::size_t row = 0; row < gate_rows; ++row)
        {
            for (std::size_t col = 0; col < gate_cols; ++col)
            {
                output(row, col) = gate_silu(row, col) * up(row, col);
            }
        }
        return output;
    }
} // namespace tinyinfer