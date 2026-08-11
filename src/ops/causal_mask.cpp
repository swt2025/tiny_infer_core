#include <cstddef>
#include <stdexcept>
#include <limits>

#include "tinyinfer/ops/causal_mask.h"

namespace tinyinfer
{
    Matrix causal_mask(const Matrix &input)
    {
        std::size_t input_rows = input.rows();
        std::size_t input_cols = input.cols();

        // 对于输入，要求行数和列数必须不为零，且相等
        if (input_rows == 0 || input_cols == 0)
        {
            throw std::invalid_argument("input Matrix cols and rows must be than zero.");
        }
        if (input_rows != input_cols)
        {
            throw std::invalid_argument("input rows must be equal with cols.");
        }

        Matrix output(input_rows, input_cols, -std::numeric_limits<float>::infinity());
        for (std::size_t row = 0; row < input_rows; ++row)
        {
            for (std::size_t col = 0; col < input_cols; ++col)
            {
                if (row >= col)
                {
                    output(row, col) = input(row, col);
                }
            }
        }
        return output;
    }
}