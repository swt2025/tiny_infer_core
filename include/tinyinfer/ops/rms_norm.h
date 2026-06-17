#pragma once


#include "tinyinfer/core/matrix.h"

namespace tinyinfer {
	Matrix RmsNorm(const Matrix& input, const Matrix& weight, float eps);
}	// namespace tinyinfer
