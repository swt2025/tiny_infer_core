#include <cmath>
#include <stdexcept>
#include "tinyinfer/ops/silu.h"

namespace tinyinfer {
    Matrix SiLU(const Matrix& input) {
        std::size_t rows = input.rows();
        std::size_t cols = input.cols();
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("SiLU input matrix must not be empty.");
        }

        Matrix output(rows, cols, 0.0f);

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                const float value = input(row, col);
                output(row, col) = value / (1.0f + std::exp(-value));
            }
        }
        return output;
    }
}