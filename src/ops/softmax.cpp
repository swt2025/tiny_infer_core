#include "tinyinfer/ops/softmax.h"

#include <cmath>

namespace tinyinfer {
	Matrix SoftmaxNaive(const Matrix& input) {
		Matrix output(input.rows(), input.cols(), 0.0f);

		for (std::size_t i = 0; i < input.rows(); ++i) { 
			float sum = 0.0f;
			for (std::size_t j = 0; j < input.cols(); ++j) {
				sum += std::exp(input(i, j));
			}
			for (std::size_t j = 0; j < input.cols(); ++j) {
				output(i, j) = std::exp(input(i, j)) / sum;
			}
		}
		
		return output;
	}

	Matrix SoftmaxStable(const Matrix& input) {
		Matrix output(input.rows(), input.cols(), 0.0f);

		for (std::size_t i = 0; i < input.rows(); ++i) {
			float max_value = input(i, 0);

			for (std::size_t j = 0; j < input.cols(); ++j) {
				if (input(i, j) > max_value) {
					max_value = input(i, j);
				}
			}

			float sum = 0.0f;
			for (std::size_t j = 0; j < input.cols(); ++j) {
				const float value = std::exp(input(i, j) - max_value);
				output(i, j) = value;
				sum += value;
			}
			
			for (std::size_t j = 0; j < input.cols(); ++j) {
				output(i, j) /= sum;
			}
		}
		
		return output;
	}
} // namespace tinyinfer

