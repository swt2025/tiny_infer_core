#include "tinyinfer/ops/rms_norm.h"

#include <cmath>
#include <stdexcept>

namespace tinyinfer {
	Matrix RmsNorm(const Matrix& input, const Matrix& weight, float eps) {
		if (input.cols() != weight.cols() || weight.rows() != 1) {
			throw std::invalid_argument("RmsNorm weight shape mismatch");
		}

		if (eps <= 0.0f) {
			throw std::invalid_argument("RmsNorm eps must be posivite");
		}

		Matrix output(input.rows(), input.cols(), 0.0);

		std::size_t hidden_size = input.cols();
		for (std::size_t row = 0; row < input.rows(); ++row) {
			float square_sum = 0.0f;
			for (std::size_t col = 0; col < hidden_size; ++col) {
				float value = input(row, col);
				square_sum += value * value;
				output(row,col) = value * weight(0,col);
			}
			for (std::size_t col = 0; col < hidden_size; ++col) { 
				const float inv_rms = 1.0f 
					/  std::sqrt(square_sum 
							/ static_cast<float>(hidden_size) + eps);
				output(row,col) = output(row,col) * inv_rms;
			}
		}	
		return output;
	}
}


