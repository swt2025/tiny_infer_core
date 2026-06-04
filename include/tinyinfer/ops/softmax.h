#pragma once
#include "tinyinfer/core/matrix.h"

namespace tinyinfer {
	Matrix SoftmaxNaive(const Matrix& input);
	Matrix SoftmaxStable(const Matrix& input);

} // namespace tinyinfer
