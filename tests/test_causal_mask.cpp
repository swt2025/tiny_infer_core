#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/causal_mask.h"

namespace
{

void Check(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void ExpectInvalidArgument(
    Function&& function,
    const std::string& expected_message)
{
    try
    {
        function();
    }
    catch (const std::invalid_argument& exception)
    {
        Check(
            exception.what() == expected_message,
            "Unexpected exception message. Expected: \"" +
                expected_message +
                "\", actual: \"" +
                exception.what() + "\"");
        return;
    }
    catch (...)
    {
        throw std::runtime_error(
            "Expected std::invalid_argument, "
            "but another exception was thrown.");
    }

    throw std::runtime_error(
        "Expected std::invalid_argument, "
        "but no exception was thrown.");
}

void TestOneByOneMatrix()
{
    tinyinfer::Matrix input(1, 1);
    input(0, 0) = 3.0f;

    const tinyinfer::Matrix output =
        tinyinfer::causal_mask(input);

    Check(output.rows() == 1, "Unexpected output row count.");
    Check(output.cols() == 1, "Unexpected output column count.");

    Check(
        output(0, 0) == 3.0f,
        "1x1 causal mask should keep the original value.");
}

void TestCausalMaskValues()
{
    tinyinfer::Matrix input(4, 4);

    float value = 1.0f;

    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t col = 0; col < 4; ++col)
        {
            input(row, col) = value++;
        }
    }

    const tinyinfer::Matrix output =
        tinyinfer::causal_mask(input);

    /*
        input:

         1   2   3   4
         5   6   7   8
         9  10  11  12
        13  14  15  16

        expected:

         1  -inf -inf -inf
         5   6   -inf -inf
         9  10   11  -inf
        13  14   15   16
    */

    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t col = 0; col < 4; ++col)
        {
            if (col <= row)
            {
                Check(
                    output(row, col) == input(row, col),
                    "Valid causal position was modified.");
            }
            else
            {
                Check(
                    std::isinf(output(row, col)),
                    "Masked position should be infinity.");

                Check(
                    output(row, col) < 0.0f,
                    "Masked position should be negative infinity.");
            }
        }
    }
}

void TestDiagonalIsPreserved()
{
    tinyinfer::Matrix input(3, 3);

    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    input(1, 0) = 4.0f;
    input(1, 1) = 5.0f;
    input(1, 2) = 6.0f;

    input(2, 0) = 7.0f;
    input(2, 1) = 8.0f;
    input(2, 2) = 9.0f;

    const tinyinfer::Matrix output =
        tinyinfer::causal_mask(input);

    Check(output(0, 0) == 1.0f, "Diagonal value changed.");
    Check(output(1, 1) == 5.0f, "Diagonal value changed.");
    Check(output(2, 2) == 9.0f, "Diagonal value changed.");
}

void TestLowerTriangleIsPreserved()
{
    tinyinfer::Matrix input(3, 3);

    float value = 1.0f;

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t col = 0; col < 3; ++col)
        {
            input(row, col) = value++;
        }
    }

    const tinyinfer::Matrix output =
        tinyinfer::causal_mask(input);

    for (std::size_t row = 0; row < 3; ++row)
    {
        for (std::size_t col = 0; col <= row; ++col)
        {
            Check(
                output(row, col) == input(row, col),
                "Lower triangular value was modified.");
        }
    }
}

void TestNonSquareMatrixThrows()
{
    tinyinfer::Matrix input(2, 3);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::causal_mask(input);
        },
        "input rows must be equal with cols.");
}

} // namespace

int main()
{
    try
    {
        TestOneByOneMatrix();
        TestCausalMaskValues();
        TestDiagonalIsPreserved();
        TestLowerTriangleIsPreserved();
        TestNonSquareMatrixThrows();

        std::cout << "All causal mask tests passed.\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Causal mask test failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}