#include <cassert>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include "tinyinfer/ops/linear.h"

namespace
{
    bool almostEqual(const float a, const float b, const float eps = 1e-5f)
    {
        return std::abs(a - b) < eps;
    }
} // namespace

void testLinearSingleRowSingleOutput()
{
    tinyinfer::Matrix input(1, 2, 0.0f);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;

    tinyinfer::Matrix weight(2, 1, 0.0f);
    weight(0, 0) = 3.0f;
    weight(1, 0) = 4.0f;

    tinyinfer::Matrix bias(1, 1, 5.0f);

    tinyinfer::Matrix output = tinyinfer::Linear(input, weight, bias);

    assert(output.rows() == 1);
    assert(output.cols() == 1);
    assert(almostEqual(output(0, 0), 16.0f));

    return;
}

void testLinearSingleRowMultipleOutput()
{
    // input: [1, 2]
    tinyinfer::Matrix input(1, 2, 0.0f);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;

    // weight: [2, 3]
    tinyinfer::Matrix weight(2, 3, 0.0f);
    weight(0, 0) = 1.0f;
    weight(0, 1) = 2.0f;
    weight(0, 2) = 3.0f;

    weight(1, 0) = 4.0f;
    weight(1, 1) = 5.0f;
    weight(1, 2) = 6.0f;

    // bias: [1, 3]
    tinyinfer::Matrix bias(1, 3, 0.0f);
    bias(0, 0) = 10.0f;
    bias(0, 1) = 20.0f;
    bias(0, 2) = 30.0f;

    tinyinfer::Matrix output = tinyinfer::Linear(input, weight, bias);

    // output[0][0] = 1*1 + 2*4 + 10 = 19
    // output[0][1] = 1*2 + 2*5 + 20 = 32
    // output[0][2] = 1*3 + 2*6 + 30 = 45
    assert(output.rows() == 1);
    assert(output.cols() == 3);

    assert(almostEqual(output(0, 0), 19.0f));
    assert(almostEqual(output(0, 1), 32.0f));
    assert(almostEqual(output(0, 2), 45.0f));
}

void testLinearInputWeightShapeMismatch()
{
    tinyinfer::Matrix input(2, 3, 1.0f);
    tinyinfer::Matrix weight(2, 2, 1.0f);
    tinyinfer::Matrix bias(1, 2, 0.0f);

    bool thrown = false;

    try
    {
        tinyinfer::Linear(input, weight, bias);
    }
    catch (const std::invalid_argument &)
    {
        thrown = true;
    }

    assert(thrown);
}

int main()
{
    testLinearSingleRowSingleOutput();
    testLinearSingleRowMultipleOutput();
    testLinearInputWeightShapeMismatch();

    std::cout << "All Linear tests passed.\n";


    return 0;
}