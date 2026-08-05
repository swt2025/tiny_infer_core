#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "tinyinfer/ops/rope.h"

namespace tinyinfer
{

Matrix RoPE(
    const Matrix& input,
    const Matrix& abs_position,
    float base)
{
    const std::size_t input_rows = input.rows();
    const std::size_t input_cols = input.cols();

    const std::size_t abs_position_rows =
        abs_position.rows();
    const std::size_t abs_position_cols =
        abs_position.cols();

    if (input_rows != abs_position_rows)
    {
        throw std::invalid_argument(
            "The number of tokens does not match "
            "the number of absolute positions.");
    }

    if (input_rows == 0 || input_cols == 0)
    {
        throw std::invalid_argument(
            "input is empty.");
    }

    if (input_cols % 2 != 0)
    {
        throw std::invalid_argument(
            "token's dimension must be even.");
    }

    if (abs_position_cols != 1)
    {
        throw std::invalid_argument(
            "absolute position's cols must be 1.");
    }

    if (base <= 0.0f)
    {
        throw std::invalid_argument(
            "RoPE base must be greater than zero.");
    }

    Matrix output(input_rows, input_cols, 0.0f);

    const std::size_t pair_count = input_cols / 2;
    Matrix theta(1, pair_count, 0.0f);

    // 预计算每个二维维度对的旋转频率。
    for (std::size_t pair = 0; pair < pair_count; ++pair)
    {
        const float exponent =
            -static_cast<float>(pair) /
            static_cast<float>(pair_count);

        theta(0, pair) =
            std::pow(base, exponent);
    }

    for (std::size_t token = 0;
         token < input_rows;
         ++token)
    {
        const float position =
            abs_position(token, 0);

        for (std::size_t pair = 0;
             pair < pair_count;
             ++pair)
        {
            const std::size_t first_col = 2 * pair;
            const std::size_t second_col = first_col + 1;

            const float value_1 =
                input(token, first_col);
            const float value_2 =
                input(token, second_col);

            const float angle =
                position * theta(0, pair);

            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);

            output(token, first_col) =
                value_1 * cosine -
                value_2 * sine;

            output(token, second_col) =
                value_1 * sine +
                value_2 * cosine;
        }
    }

    return output;
}

} // namespace tinyinfer