#include <cassert>
#include <cmath>
#include <iostream>

#include "tinyinfer/ops/silu.h"

namespace {

bool AlmostEqual(const float a, const float b, const float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void TestOneElementZero() {
    tinyinfer::Matrix input(1, 1, 0.0f);

    tinyinfer::Matrix result = tinyinfer::SiLU(input);

    assert(result.rows() == 1);
    assert(result.cols() == 1);
    assert(AlmostEqual(result(0, 0), 0.0f));
}

void TestOneElementPositive() {
    tinyinfer::Matrix input(1, 1, 1.0f);

    tinyinfer::Matrix result = tinyinfer::SiLU(input);

    const float expected = 1.0f / (1.0f + std::exp(-1.0f));

    assert(AlmostEqual(result(0, 0), expected));
}

void TestOneElementNegative() {
    tinyinfer::Matrix input(1, 1, -1.0f);

    tinyinfer::Matrix result = tinyinfer::SiLU(input);

    const float expected = -1.0f / (1.0f + std::exp(1.0f));

    assert(AlmostEqual(result(0, 0), expected));
}

}  // namespace

int main() {
    TestOneElementZero();
    TestOneElementPositive();
    TestOneElementNegative();

    std::cout << "All SiLU tests passed.\n";
    return 0;
}