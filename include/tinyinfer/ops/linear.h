#pragma once
#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"

namespace tinyinfer
{
    Matrix Linear(const Matrix &input,
                  const Matrix &weight,
                  const Matrix &bias,
                  const GemmBackend& gemm_backend = tinyinfer::GemmIKJBackend());
}