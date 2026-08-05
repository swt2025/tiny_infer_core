#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/rope.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

constexpr float kTolerance = 1e-5f;

void Check(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void CheckNear(
    float actual,
    float expected,
    const std::string& message)
{
    if (std::fabs(actual - expected) > kTolerance)
    {
        throw std::runtime_error(
            message +
            ", expected: " + std::to_string(expected) +
            ", actual: " + std::to_string(actual));
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
            "Expected std::invalid_argument, but another exception was thrown.");
    }

    throw std::runtime_error(
        "Expected std::invalid_argument, but no exception was thrown.");
}

void TestPositionZeroDoesNotChangeInput()
{
    tinyinfer::Matrix input(1, 4);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(0, 3) = 4.0f;

    tinyinfer::Matrix absolute_positions(1, 1);
    absolute_positions(0, 0) = 0.0f;

    const tinyinfer::Matrix output =
        tinyinfer::RoPE(input, absolute_positions);

    Check(output.rows() == 1, "Unexpected output row count.");
    Check(output.cols() == 4, "Unexpected output column count.");

    for (std::size_t col = 0; col < input.cols(); ++col)
    {
        CheckNear(
            output(0, col),
            input(0, col),
            "Position zero should not change the input.");
    }
}

void TestKnownRoPEValues()
{
    tinyinfer::Matrix input(1, 4);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(0, 3) = 4.0f;

    tinyinfer::Matrix absolute_positions(1, 1);
    absolute_positions(0, 0) = 1.0f;

    const tinyinfer::Matrix output =
        tinyinfer::RoPE(input, absolute_positions);

    /*
     * head_dim = 4
     *
     * Pair 0:
     * frequency = 10000^(-2 * 0 / 4) = 1
     * angle = position * frequency = 1
     *
     * Pair 1:
     * frequency = 10000^(-2 * 1 / 4) = 0.01
     * angle = position * frequency = 0.01
     */

    const float angle0 = 1.0f;
    const float expected0 =
        1.0f * std::cos(angle0) -
        2.0f * std::sin(angle0);
    const float expected1 =
        1.0f * std::sin(angle0) +
        2.0f * std::cos(angle0);

    const float angle1 = 0.01f;
    const float expected2 =
        3.0f * std::cos(angle1) -
        4.0f * std::sin(angle1);
    const float expected3 =
        3.0f * std::sin(angle1) +
        4.0f * std::cos(angle1);

    CheckNear(output(0, 0), expected0, "Incorrect value at column 0.");
    CheckNear(output(0, 1), expected1, "Incorrect value at column 1.");
    CheckNear(output(0, 2), expected2, "Incorrect value at column 2.");
    CheckNear(output(0, 3), expected3, "Incorrect value at column 3.");
}

void TestDifferentTokensUseDifferentPositions()
{
    tinyinfer::Matrix input(2, 2);

    input(0, 0) = 1.0f;
    input(0, 1) = 0.0f;

    input(1, 0) = 1.0f;
    input(1, 1) = 0.0f;

    tinyinfer::Matrix absolute_positions(2, 1);
    absolute_positions(0, 0) = 0.0f;
    absolute_positions(1, 0) = 1.0f;

    const tinyinfer::Matrix output =
        tinyinfer::RoPE(input, absolute_positions);

    // 位置 0 不旋转。
    CheckNear(output(0, 0), 1.0f, "Incorrect row 0, column 0.");
    CheckNear(output(0, 1), 0.0f, "Incorrect row 0, column 1.");

    // 位置 1 将向量 (1, 0) 旋转 1 rad。
    CheckNear(
        output(1, 0),
        std::cos(1.0f),
        "Incorrect row 1, column 0.");

    CheckNear(
        output(1, 1),
        std::sin(1.0f),
        "Incorrect row 1, column 1.");
}

void TestRotationPreservesPairLength()
{
    tinyinfer::Matrix input(1, 2);
    input(0, 0) = 3.0f;
    input(0, 1) = 4.0f;

    tinyinfer::Matrix absolute_positions(1, 1);
    absolute_positions(0, 0) = 5.0f;

    const tinyinfer::Matrix output =
        tinyinfer::RoPE(input, absolute_positions);

    const float input_squared_length =
        input(0, 0) * input(0, 0) +
        input(0, 1) * input(0, 1);

    const float output_squared_length =
        output(0, 0) * output(0, 0) +
        output(0, 1) * output(0, 1);

    CheckNear(
        output_squared_length,
        input_squared_length,
        "RoPE should preserve the length of each dimension pair.");
}

void TestTokenCountMismatchThrows()
{
    tinyinfer::Matrix input(2, 4);
    tinyinfer::Matrix absolute_positions(1, 1);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::RoPE(input, absolute_positions);
        },
        "The number of tokens does not match the number of absolute positions.");
}

void TestEmptyInputThrows()
{
    /*
     * 两个矩阵的行数必须相同，否则会优先进入
     * token 数量不匹配的异常分支。
     */
    tinyinfer::Matrix input(0, 0);
    tinyinfer::Matrix absolute_positions(0, 1);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::RoPE(input, absolute_positions);
        },
        "input is empty.");
}

void TestOddTokenDimensionThrows()
{
    tinyinfer::Matrix input(1, 3);
    tinyinfer::Matrix absolute_positions(1, 1);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::RoPE(input, absolute_positions);
        },
        "token's dimension must be even.");
}

void TestAbsolutePositionColumnCountThrows()
{
    tinyinfer::Matrix input(2, 4);
    tinyinfer::Matrix absolute_positions(2, 2);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::RoPE(input, absolute_positions);
        },
        "absolute position's cols must be 1.");
}

} // namespace

int main()
{
    try
    {
        TestPositionZeroDoesNotChangeInput();
        TestKnownRoPEValues();
        TestDifferentTokensUseDifferentPositions();
        TestRotationPreservesPairLength();

        TestTokenCountMismatchThrows();
        TestEmptyInputThrows();
        TestOddTokenDimensionThrows();
        TestAbsolutePositionColumnCountThrows();

        std::cout << "All RoPE tests passed.\n";
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "RoPE test failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}