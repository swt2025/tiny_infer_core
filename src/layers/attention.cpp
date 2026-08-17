#include <cstddef>
#include <stdexcept>
#include <cmath>

#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"
#include "tinyinfer/ops/linear.h"
#include "tinyinfer/ops/rope.h"
#include "tinyinfer/ops/transpose.h"
#include "tinyinfer/ops/causal_mask.h"
#include "tinyinfer/ops/softmax.h"
#include "tinyinfer/layers/attention.h"

namespace
{
    void CheckLinearParameters(
        const tinyinfer::Matrix &weight,
        const tinyinfer::Matrix &bias,
        const char *name)
    {
        if (bias.rows() != 1 ||
            bias.cols() != weight.cols())
        {
            throw std::invalid_argument(
                std::string(name) +
                " bias shape does not match weight.");
        }
    }
}

namespace tinyinfer
{
    Attention::Attention(
        const Matrix &q_weight, const Matrix &q_bias,
        const Matrix &k_weight, const Matrix &k_bias,
        const Matrix &v_weight, const Matrix &v_bias,
        const Matrix &o_weight, const Matrix &o_bias,
        const GemmBackend &gemm_backend,
        const float RoPE_base) : q_weight_(q_weight), q_bias_(q_bias),
                                 k_weight_(k_weight), k_bias_(k_bias),
                                 v_weight_(v_weight), v_bias_(v_bias),
                                 o_weight_(o_weight), o_bias_(o_bias),
                                 gemm_backend_(gemm_backend),
                                 RoPE_base_(RoPE_base)
    {
        CheckLinearParameters(q_weight_, q_bias_, "Q");
        CheckLinearParameters(k_weight_, k_bias_, "K");
        CheckLinearParameters(v_weight_, v_bias_, "V");
        CheckLinearParameters(o_weight_, o_bias_, "O");

        // 對於絕對位置和需要進行絕對位置編碼的進行參數檢查
        // 參與的是生成的q和k，其形狀爲input的行數、q_weight的列數進行絕對位置編碼。
        // 那麼對於絕對位置應該是一個input的行數、列數爲1的Matrix
        // 在構造函數中，應該僅檢查列數即可
        if (q_weight_.rows() != k_weight_.rows() ||
            q_weight_.rows() != v_weight_.rows())
        {
            throw std::invalid_argument(
                "Q, K and V input dimensions must match.");
        }

        if (q_weight_.cols() != k_weight_.cols() ||
            q_weight_.cols() != v_weight_.cols())
        {
            throw std::invalid_argument(
                "Q, K and V output dimensions must match.");
        }

        if (q_weight_.rows() != q_weight_.cols())
        {
            throw std::invalid_argument(
                "Single-head attention projection must preserve hidden dimension.");
        }

        if (o_weight_.rows() != q_weight_.cols() ||
            o_weight_.cols() != q_weight_.rows())
        {
            throw std::invalid_argument(
                "Output projection shape does not match attention dimension.");
        }

        if (q_weight_.cols() % 2 != 0)
        {
            throw std::invalid_argument(
                "Attention dimension must be even for RoPE.");
        }

        if (RoPE_base_ <= 0.0f)
        {
            throw std::invalid_argument(
                "RoPE base must be greater than zero.");
        }
    }

    Matrix Attention::forward(const Matrix &input, const Matrix &abs_position) const
    {
        if (input.rows() == 0 || input.cols() == 0)
        {
            throw std::invalid_argument("Attention input must not be empty.");
        }
        if (input.cols() != q_weight_.rows())
        {
            throw std::invalid_argument(
                "Attention input dimension does not match hidden dimension.");
        }
        if (abs_position.cols() != 1)
        {
            throw std::invalid_argument("abs_position input cols must be 1.");
        }

        if (input.rows() != abs_position.rows())
        {
            throw std::invalid_argument(
                "Attention input cols does not match abs_position cols.");
        }
        Matrix q = Linear(input, q_weight_, q_bias_, gemm_backend_);
        Matrix k = Linear(input, k_weight_, k_bias_, gemm_backend_);
        Matrix v = Linear(input, v_weight_, v_bias_, gemm_backend_);
        std::size_t q_rows = q.rows();
        Matrix absolute_positions = Matrix(q_rows, 1, 0.0f);
        for (std::size_t index_of_token = 0; index_of_token < q_rows; ++index_of_token)
        {
            absolute_positions(index_of_token, 0) = static_cast<float>(index_of_token);
        }

        q = RoPE(q, absolute_positions, RoPE_base_);
        k = RoPE(k, absolute_positions, RoPE_base_);

        Matrix scores = gemm_backend_.Compute(q, transpose(k));

        std::size_t scores_rows = scores.rows();
        std::size_t scores_cols = scores.cols();

        const float sqrt_nums = std::sqrt(static_cast<double>(q.cols()));

        for (std::size_t i = 0; i < scores_rows; ++i)
        {
            for (std::size_t j = 0; j < scores_cols; ++j)
            {
                scores(i, j) /= sqrt_nums;
            }
        }

        Matrix causal_score = causal_mask(scores);

        Matrix softmax_score = SoftmaxStable(causal_score);

        Matrix context = gemm_backend_.Compute(softmax_score, v);

        Matrix output = Linear(context, o_weight_, o_bias_, gemm_backend_);

        return output;
    }
}