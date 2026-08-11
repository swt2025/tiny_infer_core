#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstddef>

#include "tinyinfer/ops/transpose.h"

namespace tinyinfer {
    Matrix transpose(const Matrix& input) {
        std::size_t input_row = input.rows();
        std::size_t input_col = input.cols();
        
        // 可以接受行为0或者列为0
        Matrix output(input_col, input_row);

        if (input_col == 0 || input_row == 0) {
            return output;
        }

        for (std::size_t row = 0; row < input_row; ++row) {
            for (std::size_t col = 0; col < input_col; ++col) {
                output(col, row) = input(row, col);
            }
        }
        return output;
    }
}