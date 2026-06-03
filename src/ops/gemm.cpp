#include "tinyinfer/ops/gemm.h"

#include <stdexcept>

namespace tinyinfer {

	Matrix GemmNaive(const Matrix& a, const Matrix& b) {
		if (a.cols() != b.rows()) {
			throw std::invalid_argument("GemmNaive shape mismatch");
		}

		Matrix c(a.rows(), b.cols(), 0.0f);

		for (std::size_t i = 0; i < a.rows(); ++i) {
			for (std::size_t j = 0; j < b.cols(); ++j) {
				float sum = 0.0f;

				for (std::size_t k = 0; k < a.cols(); ++k) {
					sum += a(i, k) * b(k, j);
				}

				c(i, j) = sum;
			}
		}

		return c;
	}

}  // namespace tinyinfer
