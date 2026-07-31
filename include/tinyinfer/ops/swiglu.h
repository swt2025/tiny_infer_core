#pragma once 

#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/silu.h"

namespace tinyinfer {
    Matrix SwiGLu(const Matrix& gate, const Matrix& up);
}