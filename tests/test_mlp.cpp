#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "tinyinfer/layers/mlp.h"

namespace {

bool AlmostEqual(const float a, const float b, const float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

float SiluScalar(const float x) {
    return x / (1.0f + std::exp(-x));
}

tinyinfer::MLP CreateSimpleMlp() {
    // hidden_size = 2
    // intermediate_size = 3

    tinyinfer::Matrix gate_weight(2, 3, 0.0f);
    gate_weight(0, 0) = 1.0f;
    gate_weight(0, 1) = 0.0f;
    gate_weight(0, 2) = 1.0f;
    gate_weight(1, 0) = 0.0f;
    gate_weight(1, 1) = 1.0f;
    gate_weight(1, 2) = 1.0f;

    tinyinfer::Matrix gate_bias(1, 3, 0.0f);

    tinyinfer::Matrix up_weight(2, 3, 0.0f);
    up_weight(0, 0) = 1.0f;
    up_weight(0, 1) = 2.0f;
    up_weight(0, 2) = 0.0f;
    up_weight(1, 0) = 0.0f;
    up_weight(1, 1) = 1.0f;
    up_weight(1, 2) = 1.0f;

    tinyinfer::Matrix up_bias(1, 3, 0.0f);

    tinyinfer::Matrix down_weight(3, 2, 0.0f);
    down_weight(0, 0) = 1.0f;
    down_weight(0, 1) = 0.0f;
    down_weight(1, 0) = 0.0f;
    down_weight(1, 1) = 1.0f;
    down_weight(2, 0) = 1.0f;
    down_weight(2, 1) = 1.0f;

    tinyinfer::Matrix down_bias(1, 2, 0.0f);
    down_bias(0, 0) = 0.5f;
    down_bias(0, 1) = -0.5f;

    return tinyinfer::MLP(gate_weight,
                          gate_bias,
                          up_weight,
                          up_bias,
                          down_weight,
                          down_bias);
}

void ComputeExpectedForRow(const float x0,
                           const float x1,
                           float* expected0,
                           float* expected1) {
    // gate = input * gate_weight
    const float gate0 = x0;
    const float gate1 = x1;
    const float gate2 = x0 + x1;

    // up = input * up_weight
    const float up0 = x0;
    const float up1 = 2.0f * x0 + x1;
    const float up2 = x1;

    // hidden = SwiGLU(gate, up) = SiLU(gate) * up
    const float hidden0 = SiluScalar(gate0) * up0;
    const float hidden1 = SiluScalar(gate1) * up1;
    const float hidden2 = SiluScalar(gate2) * up2;

    // output = hidden * down_weight + down_bias
    *expected0 = hidden0 + hidden2 + 0.5f;
    *expected1 = hidden1 + hidden2 - 0.5f;
}

void TestMlpSingleRow() {
    tinyinfer::MLP mlp = CreateSimpleMlp();

    tinyinfer::Matrix input(1, 2, 0.0f);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;

    tinyinfer::Matrix output = mlp.forward(input);

    float expected0 = 0.0f;
    float expected1 = 0.0f;
    ComputeExpectedForRow(1.0f, 2.0f, &expected0, &expected1);

    assert(output.rows() == 1);
    assert(output.cols() == 2);
    assert(AlmostEqual(output(0, 0), expected0));
    assert(AlmostEqual(output(0, 1), expected1));
}

void TestMlpMultipleRows() {
    tinyinfer::MLP mlp = CreateSimpleMlp();

    tinyinfer::Matrix input(2, 2, 0.0f);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(1, 0) = 3.0f;
    input(1, 1) = 4.0f;

    tinyinfer::Matrix output = mlp.forward(input);

    float expected00 = 0.0f;
    float expected01 = 0.0f;
    float expected10 = 0.0f;
    float expected11 = 0.0f;

    ComputeExpectedForRow(1.0f, 2.0f, &expected00, &expected01);
    ComputeExpectedForRow(3.0f, 4.0f, &expected10, &expected11);

    assert(output.rows() == 2);
    assert(output.cols() == 2);

    assert(AlmostEqual(output(0, 0), expected00));
    assert(AlmostEqual(output(0, 1), expected01));
    assert(AlmostEqual(output(1, 0), expected10));
    assert(AlmostEqual(output(1, 1), expected11));
}

void TestMlpGateUpWeightRowsMismatch() {
    tinyinfer::Matrix gate_weight(2, 3, 1.0f);
    tinyinfer::Matrix gate_bias(1, 3, 0.0f);

    tinyinfer::Matrix up_weight(3, 3, 1.0f);
    tinyinfer::Matrix up_bias(1, 3, 0.0f);

    tinyinfer::Matrix down_weight(3, 2, 1.0f);
    tinyinfer::Matrix down_bias(1, 2, 0.0f);

    bool thrown = false;

    try {
        tinyinfer::MLP mlp(gate_weight,
                           gate_bias,
                           up_weight,
                           up_bias,
                           down_weight,
                           down_bias);
        (void)mlp;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestMlpGateBiasShapeMismatch() {
    tinyinfer::Matrix gate_weight(2, 3, 1.0f);
    tinyinfer::Matrix gate_bias(1, 2, 0.0f);

    tinyinfer::Matrix up_weight(2, 3, 1.0f);
    tinyinfer::Matrix up_bias(1, 3, 0.0f);

    tinyinfer::Matrix down_weight(3, 2, 1.0f);
    tinyinfer::Matrix down_bias(1, 2, 0.0f);

    bool thrown = false;

    try {
        tinyinfer::MLP mlp(gate_weight,
                           gate_bias,
                           up_weight,
                           up_bias,
                           down_weight,
                           down_bias);
        (void)mlp;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestMlpDownWeightInputMismatch() {
    tinyinfer::Matrix gate_weight(2, 3, 1.0f);
    tinyinfer::Matrix gate_bias(1, 3, 0.0f);

    tinyinfer::Matrix up_weight(2, 3, 1.0f);
    tinyinfer::Matrix up_bias(1, 3, 0.0f);

    // down_weight.rows() 应该等于 intermediate_size，也就是 3。
    // 这里故意写成 4。
    tinyinfer::Matrix down_weight(4, 2, 1.0f);
    tinyinfer::Matrix down_bias(1, 2, 0.0f);

    bool thrown = false;

    try {
        tinyinfer::MLP mlp(gate_weight,
                           gate_bias,
                           up_weight,
                           up_bias,
                           down_weight,
                           down_bias);
        (void)mlp;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestMlpDownWeightOutputMismatch() {
    tinyinfer::Matrix gate_weight(2, 3, 1.0f);
    tinyinfer::Matrix gate_bias(1, 3, 0.0f);

    tinyinfer::Matrix up_weight(2, 3, 1.0f);
    tinyinfer::Matrix up_bias(1, 3, 0.0f);

    // down_weight.cols() 应该等于 hidden_size，也就是 2。
    // 这里故意写成 4。
    tinyinfer::Matrix down_weight(3, 4, 1.0f);
    tinyinfer::Matrix down_bias(1, 4, 0.0f);

    bool thrown = false;

    try {
        tinyinfer::MLP mlp(gate_weight,
                           gate_bias,
                           up_weight,
                           up_bias,
                           down_weight,
                           down_bias);
        (void)mlp;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestMlpForwardInputShapeMismatch() {
    tinyinfer::MLP mlp = CreateSimpleMlp();

    // MLP hidden_size = 2，但这里 input.cols() = 3。
    tinyinfer::Matrix input(1, 3, 1.0f);

    bool thrown = false;

    try {
        tinyinfer::Matrix output = mlp.forward(input);
        (void)output;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void TestMlpForwardEmptyInput() {
    tinyinfer::MLP mlp = CreateSimpleMlp();

    tinyinfer::Matrix input(0, 2, 0.0f);

    bool thrown = false;

    try {
        tinyinfer::Matrix output = mlp.forward(input);
        (void)output;
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

}  // namespace

int main() {
    TestMlpSingleRow();
    TestMlpMultipleRows();

    TestMlpGateUpWeightRowsMismatch();
    TestMlpGateBiasShapeMismatch();
    TestMlpDownWeightInputMismatch();
    TestMlpDownWeightOutputMismatch();

    TestMlpForwardInputShapeMismatch();
    TestMlpForwardEmptyInput();

    std::cout << "All MLP tests passed.\n";
    return 0;
}