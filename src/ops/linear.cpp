#include "tinyinfer/ops/linear.h"

#include <cstddef>
#include <stdexcept>

#include "tinyinfer/ops/gemm.h"

namespace tinyinfer
{

    Matrix Linear(const Matrix &input,
                  const Matrix &weight,
                  const Matrix &bias,
                  const GemmBackend& gemm_backend)
    {
        if (input.rows() == 0 || input.cols() == 0 ||
            weight.rows() == 0 || weight.cols() == 0 ||
            bias.rows() == 0 || bias.cols() == 0)
        {
            throw std::invalid_argument("Linear input matrices must not be empty.");
        }

        if (input.cols() != weight.rows())
        {
            throw std::invalid_argument("Linear input and weight shapes are not compatible.");
        }

        if (bias.rows() != 1 || bias.cols() != weight.cols())
        {
            throw std::invalid_argument("Linear bias shape must be [1, out_features].");
        }

        Matrix output = gemm_backend.Compute(input, weight);

        for (std::size_t row = 0; row < output.rows(); ++row)
        {
            for (std::size_t col = 0; col < output.cols(); ++col)
            {
                output(row, col) += bias(0, col);
            }
        }

        return output;
    }

} // namespace tinyinfer