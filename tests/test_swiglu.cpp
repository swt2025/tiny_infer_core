#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tinyinfer/ops/swiglu.h"

namespace {

bool AlmostEqual(const float a, const float b, const float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

float SiluScalar(const float x) {
    return x / (1.0f + std::exp(-x));
}

void TestSwiGLuZeroGate() {
    tinyinfer::Matrix gate(1, 1, 0.0f);
    tinyinfer::Matrix up(1, 1, 2.0f);

    tinyinfer::Matrix result = tinyinfer::SwiGLu(gate, up);

    assert(result.rows() == 1);
    assert(result.cols() == 1);

    // SiLU(0) = 0，所以 0 * 2 = 0
    assert(AlmostEqual(result(0, 0), 0.0f));
}

void TestSwiGLuPositiveGate() {
    tinyinfer::Matrix gate(1, 1, 1.0f);
    tinyinfer::Matrix up(1, 1, 3.0f);

    tinyinfer::Matrix result = tinyinfer::SwiGLu(gate, up);

    const float expected = SiluScalar(1.0f) * 3.0f;

    assert(result.rows() == 1);
    assert(result.cols() == 1);
    assert(AlmostEqual(result(0, 0), expected));
}

void TestSwiGLuNegativeGate() {
    tinyinfer::Matrix gate(1, 1, -1.0f);
    tinyinfer::Matrix up(1, 1, 4.0f);

    tinyinfer::Matrix result = tinyinfer::SwiGLu(gate, up);

    const float expected = SiluScalar(-1.0f) * 4.0f;

    assert(result.rows() == 1);
    assert(result.cols() == 1);
    assert(AlmostEqual(result(0, 0), expected));
}

void TestSwiGLuMultiElementMatrix() {
    tinyinfer::Matrix gate(2, 3, 0.0f);
    tinyinfer::Matrix up(2, 3, 0.0f);

    gate(0, 0) = 0.0f;
    gate(0, 1) = 1.0f;
    gate(0, 2) = -1.0f;
    gate(1, 0) = 2.0f;
    gate(1, 1) = -2.0f;
    gate(1, 2) = 0.5f;

    up(0, 0) = 1.0f;
    up(0, 1) = 2.0f;
    up(0, 2) = 3.0f;
    up(1, 0) = 4.0f;
    up(1, 1) = 5.0f;
    up(1, 2) = 6.0f;

    tinyinfer::Matrix result = tinyinfer::SwiGLu(gate, up);

    assert(result.rows() == 2);
    assert(result.cols() == 3);

    assert(AlmostEqual(result(0, 0), SiluScalar(0.0f) * 1.0f));
    assert(AlmostEqual(result(0, 1), SiluScalar(1.0f) * 2.0f));
    assert(AlmostEqual(result(0, 2), SiluScalar(-1.0f) * 3.0f));
    assert(AlmostEqual(result(1, 0), SiluScalar(2.0f) * 4.0f));
    assert(AlmostEqual(result(1, 1), SiluScalar(-2.0f) * 5.0f));
    assert(AlmostEqual(result(1, 2), SiluScalar(0.5f) * 6.0f));
}

void TestSwiGLuShapeMismatch() {
    tinyinfer::Matrix gate(2, 3, 1.0f);
    tinyinfer::Matrix up(2, 2, 1.0f);

    bool thrown = false;

    try {
        tinyinfer::SwiGLu(gate, up);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestSwiGLuEmptyInput() {
    tinyinfer::Matrix gate(0, 3, 1.0f);
    tinyinfer::Matrix up(0, 3, 1.0f);

    bool thrown = false;

    try {
        tinyinfer::SwiGLu(gate, up);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

}  // namespace

int main() {
    TestSwiGLuZeroGate();
    TestSwiGLuPositiveGate();
    TestSwiGLuNegativeGate();
    TestSwiGLuMultiElementMatrix();
    TestSwiGLuShapeMismatch();
    TestSwiGLuEmptyInput();

    std::cout << "All SwiGLU tests passed.\n";
    return 0;
}