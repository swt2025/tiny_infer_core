#pragma once

#include "tinyinfer/core/matrix.h"

namespace tinyinfer {

	Matrix GemmNaive(const Matrix& a, const Matrix& b);
	Matrix GemmIJK(const Matrix& a, const Matrix& b);
	Matrix GemmIKJ(const Matrix& a, const Matrix& b);

	Matrix GemmBlocked(const Matrix& a, const Matrix& b, std::size_t block_size);

}  // namespace tinyinfer
