#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/rms_norm.h"

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <cmath>

namespace {
	bool AlmostEqual(float a, float b, float eps = 1e-5f) {
		return std::abs(a - b) < eps;
	}

	void TestRmsNormSimpleCase() {
		tinyinfer::Matrix input(1, 4);
		tinyinfer::Matrix weight(1, 4, 1.0f);

		input(0,0) = 1.0f;
		input(0, 1) = 2.0f;
		input(0, 2) = 3.0f;
		input(0, 3) = 4.0f;

		const float eps = 1e-6f;
		tinyinfer::Matrix output = tinyinfer::RmsNorm(input, weight, eps);
		const float square_sum = 1.0f * 1.0f +
			2.0f * 2.0f +
			3.0f * 3.0f +
			4.0f * 4.0f;
		
		const float mean_square = square_sum / 4.0f;
		const float inv_rms = 1.0f / std::sqrt(mean_square + eps);

		assert(AlmostEqual(output(0,0), 1.0f * inv_rms));
		assert(AlmostEqual(output(0,1), 2.0f * inv_rms));
		assert(AlmostEqual(output(0,2), 3.0f * inv_rms));
		assert(AlmostEqual(output(0,3), 4.0f * inv_rms));
	}

void TestRmsNormWithWeight() {
    tinyinfer::Matrix input(1, 3);
    tinyinfer::Matrix weight(1, 3);

    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 2.0f;

    weight(0, 0) = 1.0f;
    weight(0, 1) = 2.0f;
    weight(0, 2) = 3.0f;

    const float eps = 1e-6f;
    tinyinfer::Matrix output = tinyinfer::RmsNorm(input, weight, eps);

    const float square_sum = 1.0f + 4.0f + 4.0f;
    const float mean_square = square_sum / 3.0f;
    const float inv_rms = 1.0f / std::sqrt(mean_square + eps);

    assert(AlmostEqual(output(0, 0), 1.0f * inv_rms * 1.0f));
    assert(AlmostEqual(output(0, 1), 2.0f * inv_rms * 2.0f));
    assert(AlmostEqual(output(0, 2), 2.0f * inv_rms * 3.0f));
}

void TestRmsNormMultipleRows() {
    tinyinfer::Matrix input(2, 2);
    tinyinfer::Matrix weight(1, 2, 1.0f);

    input(0, 0) = 3.0f;
    input(0, 1) = 4.0f;

    input(1, 0) = 5.0f;
    input(1, 1) = 12.0f;

    tinyinfer::Matrix output = tinyinfer::RmsNorm(input, weight, 1e-6f);

    assert(output.rows() == 2);
    assert(output.cols() == 2);

    for (std::size_t i = 0; i < output.rows(); ++i) {
        for (std::size_t j = 0; j < output.cols(); ++j) {
            assert(std::isfinite(output(i, j)));
        }
    }
}

void TestRmsNormWeightShapeMismatch() {
    tinyinfer::Matrix input(2, 4);
    tinyinfer::Matrix weight(1, 3);

    bool caught = false;

    try {
        tinyinfer::Matrix output = tinyinfer::RmsNorm(input, weight, 1e-6f);
        (void)output;
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    assert(caught);
}

void TestRmsNormInvalidEps() {
    tinyinfer::Matrix input(1, 4);
    tinyinfer::Matrix weight(1, 4);

    bool caught = false;

    try {
        tinyinfer::Matrix output = tinyinfer::RmsNorm(input, weight, 0.0f);
        (void)output;
    } catch (const std::invalid_argument&) {
        caught = true;
    }

    assert(caught);
}

}  // namespace

int main() {
    TestRmsNormSimpleCase();
    TestRmsNormWithWeight();
    TestRmsNormMultipleRows();
    TestRmsNormWeightShapeMismatch();
    TestRmsNormInvalidEps();

    std::cout << "All RMSNorm tests passed.\n";
    return 0;
}
