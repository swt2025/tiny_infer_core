#pragma once
#include "tinyinfer/core/matrix.h"

namespace tinyinfer {
    Matrix Linear(const Matrix& input, const Matrix& weight, const Matrix& bias);
}