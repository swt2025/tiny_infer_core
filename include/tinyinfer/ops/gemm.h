#pragma once

#include "tinyinfer/core/matrix.h"

namespace tinyinfer {

	Matrix GemmNaive(const Matrix& a, const Matrix& b);
	Matrix GemmIJK(const Matrix& a, const Matrix& b);
	Matrix GemmIKJ(const Matrix& a, const Matrix& b);

}  // namespace tinyinfer
