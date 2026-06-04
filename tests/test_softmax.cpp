#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/softmax.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {
	bool AlmostEqual(float a, float b, float eps = 1e-5f) {
		return std::abs(a - b) < eps;
	}

	void TestSoftmaxRowSumToOne() {
		tinyinfer::Matrix input(2, 3);

		input(0, 0) = 1.0f;
		input(0, 1) = 2.0f;
		input(0, 2) = 3.0f;

		input(1, 0) = 4.0f;
		input(1, 1) = 5.0f;
		input(1, 2) = 6.0f;

		tinyinfer::Matrix output = tinyinfer::SoftmaxStable(input);

		for (std::size_t i = 0; i < output.rows(); ++i) {
			float sum = 0.0f;

			for (std::size_t j = 0; j < output.cols(); ++j) {
				sum += output(i, j);
			}

			assert(AlmostEqual(sum, 1.0f));
		}
	}

	void TestSoftmaxNaiveEqualsStableForSmallValues() {
		tinyinfer::Matrix input(1, 3);

		input(0, 0) = 1.0f;
		input(0, 1) = 2.0f;
		input(0, 2) = 3.0f;

		tinyinfer::Matrix naive = tinyinfer::SoftmaxNaive(input);
		tinyinfer::Matrix stable = tinyinfer::SoftmaxStable(input);

		for (std::size_t j = 0; j < input.cols(); ++j) {
			assert(AlmostEqual(naive(0, j), stable(0, j)));
		}
	}

	void TestSoftmaxLargeValues() {
		tinyinfer::Matrix input(1, 3);

		input(0, 0) = 1000.0f;
		input(0, 1) = 1001.0f;
		input(0, 2) = 1002.0f;

		tinyinfer::Matrix output = tinyinfer::SoftmaxStable(input);

		float sum = 0.0f;

		for (std::size_t j = 0; j < output.cols(); ++j) {
			assert(std::isfinite(output(0, j)));
			sum += output(0, j);
		}

		assert(AlmostEqual(sum, 1.0f));
		assert(output(0, 2) > output(0, 1));
		assert(output(0, 1) > output(0, 0));
	}

	void TestSoftmaxSameValues() {
		tinyinfer::Matrix input(1, 4, 5.0f);

		tinyinfer::Matrix output = tinyinfer::SoftmaxStable(input);

		for (std::size_t j = 0; j < output.cols(); ++j) {
			assert(AlmostEqual(output(0, j), 0.25f));
		}
	}
} // namespace
int main() {
	TestSoftmaxRowSumToOne();
	TestSoftmaxNaiveEqualsStableForSmallValues();
	TestSoftmaxLargeValues();
	TestSoftmaxSameValues();

	std::cout << "All Softmax tests passed.\n";
	return 0;
}
