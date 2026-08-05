# pragma once

#include "tinyinfer/core/matrix.h"

namespace tinyinfer 
{
    Matrix RoPE(const Matrix& input, const Matrix& abs_position, float base = 10000.0f);    
} // namespace
