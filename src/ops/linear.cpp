#include <cstddef>
#include <stdexcept>
#include "tinyinfer/ops/linear.h"
#include "tinyinfer/ops/gemm.h"

namespace tinyinfer {
    Matrix Linear(const Matrix& input, const Matrix& weight, const Matrix& bias) {
        std::size_t input_rows = input.rows();
        std::size_t input_cols = input.cols();
        std::size_t weight_rows = weight.rows();
        std::size_t weight_cols = weight.cols();
        std::size_t bias_rows = bias.rows();
        std::size_t bias_cols = bias.cols();

        if (input_rows != 1 || bias_rows != 1) {
            throw std::invalid_argument("input and bias must be a vector!");
        }

        if (input_cols != weight_rows || weight_cols != bias_cols) {
            throw std::invalid_argument("dimension not match!");
        }
        
        Matrix output = Gemm(input, weight);

        for (std::size_t row = 0; row < bias_rows; ++row) {
            for (std::size_t col = 0; col < bias_cols; ++col) {
                output(row, col) += bias(row, col);
            }
        }

        return output;
    }
}