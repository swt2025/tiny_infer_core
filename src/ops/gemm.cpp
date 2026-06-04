#include "tinyinfer/ops/gemm.h"

#include <stdexcept>

namespace tinyinfer {
	void CheckGemmShape(const Matrix& a, const Matrix& b) {
		if (a.cols() != b.rows()) {
			throw std::invalid_argument("GemmNaive shape mismatch");
		}
	}

	Matrix GemmIJK(const Matrix& a, const Matrix& b) {
		CheckGemmShape(a, b);

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

	Matrix GemmIKJ(const Matrix& a, const Matrix& b) {
		CheckGemmShape(a, b);

		Matrix c(a.rows(), b.cols(), 0.0f);

		for (std::size_t i = 0; i < a.rows(); ++i) {
			for (std::size_t k = 0; k < a.cols(); ++k) {
				const float a_value = a(i, k);
				for (std::size_t j = 0; j < b.cols(); ++j) {
					c(i, j) += a_value * b(k, j);
				}
			}
		}

		return c;
	}

	Matrix GemmNaive(const Matrix& a, const Matrix& b) {
		return		GemmIJK(a, b);
	}

	Matrix GemmBlocked(const Matrix& a, const Matrix& b, std::size_t block_size) {
		CheckGemmShape(a, b);
		if (block_size == 0) {
			throw std::invalid_argument("block_size must be greater than 0");
		}

		Matrix c(a.rows(), b.cols(), 0.0f);

		const std::size_t m = a.rows();
		const std::size_t n = b.cols();
		const std::size_t k_size = a.cols();

		for (std::size_t ii = 0; ii < m; ii += block_size) {
			for (std::size_t kk = 0; kk < k_size; kk += block_size) {
				for (std::size_t jj = 0; jj < n; jj += block_size) {
					const std::size_t i_end = std::min(ii + block_size, m);
					const std::size_t k_end = std::min(kk + block_size, k_size);
					const std::size_t j_end = std::min(jj + block_size, n);


					for (std::size_t i = ii; i < i_end; ++i) {
						for (std::size_t k = kk; k < k_end; ++k) {
							const float a_value = a(i, k);
							for (std::size_t j = jj; j < j_end; ++j) {
								c(i, j) += a_value * b(k, j);
							}
						}
					}
				}
			}
		}
		return c;
	}

}  // namespace tinyinfer
