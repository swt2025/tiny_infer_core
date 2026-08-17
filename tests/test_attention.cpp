#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tinyinfer/core/matrix.h"
#include "tinyinfer/layers/attention.h"
#include "tinyinfer/ops/gemm.h"

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
            "Expected std::invalid_argument, "
            "but another exception was thrown.");
    }

    throw std::runtime_error(
        "Expected std::invalid_argument, "
        "but no exception was thrown.");
}

tinyinfer::Matrix MakeZeroWeight()
{
    return tinyinfer::Matrix(2, 2, 0.0f);
}

tinyinfer::Matrix MakeIdentityWeight()
{
    tinyinfer::Matrix weight(2, 2, 0.0f);

    weight(0, 0) = 1.0f;
    weight(1, 1) = 1.0f;

    return weight;
}

tinyinfer::Matrix MakeZeroBias()
{
    return tinyinfer::Matrix(1, 2, 0.0f);
}

//
// 1. 单 token
//
void TestSingleTokenAttention()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix zero_weight =
        MakeZeroWeight();

    const tinyinfer::Matrix identity_weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix zero_bias =
        MakeZeroBias();

    /*
        Q = 0
        K = 0

        score = [0]
        causal mask 不改变
        softmax([0]) = [1]

        V = input
        context = V
        Wo = identity

        所以 output == input
    */

    tinyinfer::Attention attention(
        zero_weight, zero_bias,
        zero_weight, zero_bias,
        identity_weight, zero_bias,
        identity_weight, zero_bias,
        backend,
        10000.0f);

    tinyinfer::Matrix input(1, 2);

    input(0, 0) = 2.0f;
    input(0, 1) = 4.0f;

    tinyinfer::Matrix abs_position(1, 1);
    abs_position(0, 0) = 0.0f;

    const tinyinfer::Matrix output =
        attention.forward(input, abs_position);

    Check(
        output.rows() == 1,
        "Unexpected output row count.");

    Check(
        output.cols() == 2,
        "Unexpected output column count.");

    CheckNear(
        output(0, 0),
        2.0f,
        "Incorrect single-token output column 0.");

    CheckNear(
        output(0, 1),
        4.0f,
        "Incorrect single-token output column 1.");
}

//
// 2. 两个 token，测试 causal attention
//
void TestTwoTokenCausalAttention()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix zero_weight =
        MakeZeroWeight();

    const tinyinfer::Matrix identity_weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix zero_bias =
        MakeZeroBias();

    /*
        Q = 0
        K = 0

        scores:
        [0 0
         0 0]

        causal mask:
        [0 -inf
         0    0]

        softmax:
        [1   0
         0.5 0.5]

        V = input:

        [2 4
         6 8]

        context:

        token0:
        [2, 4]

        token1:
        0.5*[2,4] + 0.5*[6,8]
        = [4,6]

        Wo = identity
    */

    tinyinfer::Attention attention(
        zero_weight, zero_bias,
        zero_weight, zero_bias,
        identity_weight, zero_bias,
        identity_weight, zero_bias,
        backend,
        10000.0f);

    tinyinfer::Matrix input(2, 2);

    input(0, 0) = 2.0f;
    input(0, 1) = 4.0f;

    input(1, 0) = 6.0f;
    input(1, 1) = 8.0f;

    tinyinfer::Matrix abs_position(2, 1);

    abs_position(0, 0) = 0.0f;
    abs_position(1, 0) = 1.0f;

    const tinyinfer::Matrix output =
        attention.forward(input, abs_position);

    CheckNear(
        output(0, 0),
        2.0f,
        "Token 0 column 0 is incorrect.");

    CheckNear(
        output(0, 1),
        4.0f,
        "Token 0 column 1 is incorrect.");

    CheckNear(
        output(1, 0),
        4.0f,
        "Token 1 column 0 is incorrect.");

    CheckNear(
        output(1, 1),
        6.0f,
        "Token 1 column 1 is incorrect.");
}

//
// 3. Output projection
//
void TestOutputProjection()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix zero_weight =
        MakeZeroWeight();

    const tinyinfer::Matrix identity_weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix zero_bias =
        MakeZeroBias();

    tinyinfer::Matrix output_weight(
        2,
        2,
        0.0f);

    output_weight(0, 0) = 2.0f;
    output_weight(1, 1) = 3.0f;

    tinyinfer::Attention attention(
        zero_weight, zero_bias,
        zero_weight, zero_bias,
        identity_weight, zero_bias,
        output_weight, zero_bias,
        backend,
        10000.0f);

    tinyinfer::Matrix input(1, 2);

    input(0, 0) = 2.0f;
    input(0, 1) = 4.0f;

    tinyinfer::Matrix abs_position(1, 1);
    abs_position(0, 0) = 0.0f;

    const tinyinfer::Matrix output =
        attention.forward(input, abs_position);

    /*
        context = [2,4]

        Wo =
        [2 0
         0 3]

        output = [4,12]
    */

    CheckNear(
        output(0, 0),
        4.0f,
        "Output projection column 0 is incorrect.");

    CheckNear(
        output(0, 1),
        12.0f,
        "Output projection column 1 is incorrect.");
}

//
// 4. forward 输入维度错误
//
void TestInputDimensionMismatchThrows()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix bias =
        MakeZeroBias();

    tinyinfer::Attention attention(
        weight, bias,
        weight, bias,
        weight, bias,
        weight, bias,
        backend);

    tinyinfer::Matrix input(
        1,
        4,
        0.0f);

    tinyinfer::Matrix abs_position(
        1,
        1,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            attention.forward(
                input,
                abs_position);
        },
        "Attention input dimension does not match hidden dimension.");
}

//
// 5. abs_position 列数错误
//
void TestAbsolutePositionColumnMismatchThrows()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix bias =
        MakeZeroBias();

    tinyinfer::Attention attention(
        weight, bias,
        weight, bias,
        weight, bias,
        weight, bias,
        backend);

    tinyinfer::Matrix input(
        2,
        2,
        0.0f);

    tinyinfer::Matrix abs_position(
        2,
        2,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            attention.forward(
                input,
                abs_position);
        },
        "abs_position input cols must be 1.");
}

//
// 6. token 数和位置数不一致
//
void TestAbsolutePositionRowMismatchThrows()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix bias =
        MakeZeroBias();

    tinyinfer::Attention attention(
        weight, bias,
        weight, bias,
        weight, bias,
        weight, bias,
        backend);

    tinyinfer::Matrix input(
        2,
        2,
        0.0f);

    tinyinfer::Matrix abs_position(
        1,
        1,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            attention.forward(
                input,
                abs_position);
        },
        "Attention input cols does not match abs_position cols.");
}

//
// 7. Q bias shape 错误
//
void TestQBiasMismatchThrows()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix correct_bias =
        MakeZeroBias();

    tinyinfer::Matrix wrong_bias(
        1,
        3,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::Attention attention(
                weight, wrong_bias,
                weight, correct_bias,
                weight, correct_bias,
                weight, correct_bias,
                backend);
        },
        "Q bias shape does not match weight.");
}

//
// 8. Q/K/V input dimension 不一致
//
void TestQKVInputDimensionMismatchThrows()
{
    tinyinfer::GemmIKJBackend backend;

    tinyinfer::Matrix q_weight(
        2,
        2,
        0.0f);

    tinyinfer::Matrix k_weight(
        4,
        2,
        0.0f);

    tinyinfer::Matrix v_weight(
        2,
        2,
        0.0f);

    tinyinfer::Matrix bias(
        1,
        2,
        0.0f);

    tinyinfer::Matrix o_weight(
        2,
        2,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::Attention attention(
                q_weight, bias,
                k_weight, bias,
                v_weight, bias,
                o_weight, bias,
                backend);
        },
        "Q, K and V input dimensions must match.");
}

//
// 9. 奇数 attention dimension
//
void TestOddAttentionDimensionThrows()
{
    tinyinfer::GemmIKJBackend backend;

    tinyinfer::Matrix weight(
        3,
        3,
        0.0f);

    tinyinfer::Matrix bias(
        1,
        3,
        0.0f);

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::Attention attention(
                weight, bias,
                weight, bias,
                weight, bias,
                weight, bias,
                backend);
        },
        "Attention dimension must be even for RoPE.");
}

//
// 10. 非法 RoPE base
//
void TestInvalidRoPEBaseThrows()
{
    tinyinfer::GemmIKJBackend backend;

    const tinyinfer::Matrix weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix bias =
        MakeZeroBias();

    ExpectInvalidArgument(
        [&]()
        {
            tinyinfer::Attention attention(
                weight, bias,
                weight, bias,
                weight, bias,
                weight, bias,
                backend,
                0.0f);
        },
        "RoPE base must be greater than zero.");
}

//
// 11. 验证传入的 abs_position 真正生效
//
void TestAbsolutePositionActuallyAffectsAttention()
{
    tinyinfer::GemmIKJBackend backend;

    /*
        这里不能让 Q/K 为零，
        否则 RoPE 无论位置是多少都不会影响结果。
    */

    const tinyinfer::Matrix identity_weight =
        MakeIdentityWeight();

    const tinyinfer::Matrix zero_bias =
        MakeZeroBias();

    tinyinfer::Attention attention(
        identity_weight, zero_bias,
        identity_weight, zero_bias,
        identity_weight, zero_bias,
        identity_weight, zero_bias,
        backend);

    tinyinfer::Matrix input(2, 2);

    input(0, 0) = 1.0f;
    input(0, 1) = 0.0f;

    input(1, 0) = 0.0f;
    input(1, 1) = 1.0f;

    tinyinfer::Matrix positions_a(2, 1);

    positions_a(0, 0) = 0.0f;
    positions_a(1, 0) = 1.0f;

    tinyinfer::Matrix positions_b(2, 1);

    positions_b(0, 0) = 10.0f;
    positions_b(1, 0) = 20.0f;

    const tinyinfer::Matrix output_a =
        attention.forward(
            input,
            positions_a);

    const tinyinfer::Matrix output_b =
        attention.forward(
            input,
            positions_b);

    bool any_difference = false;

    for (std::size_t row = 0;
         row < output_a.rows();
         ++row)
    {
        for (std::size_t col = 0;
             col < output_a.cols();
             ++col)
        {
            if (std::fabs(
                    output_a(row, col) -
                    output_b(row, col)) >
                kTolerance)
            {
                any_difference = true;
            }
        }
    }

    Check(
        any_difference,
        "Different absolute positions should affect attention output.");
}

} // namespace

int main()
{
    try
    {
        TestSingleTokenAttention();
        TestTwoTokenCausalAttention();
        TestOutputProjection();

        TestInputDimensionMismatchThrows();
        TestAbsolutePositionColumnMismatchThrows();
        TestAbsolutePositionRowMismatchThrows();

        TestQBiasMismatchThrows();
        TestQKVInputDimensionMismatchThrows();
        TestOddAttentionDimensionThrows();
        TestInvalidRoPEBaseThrows();

        TestAbsolutePositionActuallyAffectsAttention();

        std::cout
            << "All attention tests passed.\n";

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr
            << "Attention test failed: "
            << exception.what()
            << '\n';

        return 1;
    }
}