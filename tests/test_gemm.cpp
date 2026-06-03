#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

	bool AlmostEqual(float a, float b, float eps = 1e-5f) {
		return std::fabs(a - b) < eps;
	}

	void TestGemmSmallFixed() {
		tinyinfer::Matrix a(2, 3);
		tinyinfer::Matrix b(3, 2);

		a(0, 0) = 1.0f;
		a(0, 1) = 2.0f;
		a(0, 2) = 3.0f;
		a(1, 0) = 4.0f;
		a(1, 1) = 5.0f;
		a(1, 2) = 6.0f;

		b(0, 0) = 7.0f;
		b(0, 1) = 8.0f;
		b(1, 0) = 9.0f;
		b(1, 1) = 10.0f;
		b(2, 0) = 11.0f;
		b(2, 1) = 12.0f;

		tinyinfer::Matrix c = tinyinfer::GemmNaive(a, b);

		assert(c.rows() == 2);
		assert(c.cols() == 2);

		assert(AlmostEqual(c(0, 0), 58.0f));
		assert(AlmostEqual(c(0, 1), 64.0f));
		assert(AlmostEqual(c(1, 0), 139.0f));
		assert(AlmostEqual(c(1, 1), 154.0f));
	}

	void TestGemmIdentity() {
		tinyinfer::Matrix a(2, 2);
		tinyinfer::Matrix identity(2, 2);

		a(0, 0) = 1.0f;
		a(0, 1) = 2.0f;
		a(1, 0) = 3.0f;
		a(1, 1) = 4.0f;

		identity(0, 0) = 1.0f;
		identity(0, 1) = 0.0f;
		identity(1, 0) = 0.0f;
		identity(1, 1) = 1.0f;

		tinyinfer::Matrix c = tinyinfer::GemmNaive(a, identity);

		assert(AlmostEqual(c(0, 0), 1.0f));
		assert(AlmostEqual(c(0, 1), 2.0f));
		assert(AlmostEqual(c(1, 0), 3.0f));
		assert(AlmostEqual(c(1, 1), 4.0f));
	}

	void TestGemmZeroMatrix() {
		tinyinfer::Matrix a(2, 3, 1.0f);
		tinyinfer::Matrix b(3, 2, 0.0f);

		tinyinfer::Matrix c = tinyinfer::GemmNaive(a, b);

		for (std::size_t i = 0; i < c.rows(); ++i) {
			for (std::size_t j = 0; j < c.cols(); ++j) {
				assert(AlmostEqual(c(i, j), 0.0f));
			}
		}
	}

	void TestGemmShapeMismatch() {
		tinyinfer::Matrix a(2, 3);
		tinyinfer::Matrix b(4, 2);

		bool caught = false;

		try {
			tinyinfer::Matrix c = tinyinfer::GemmNaive(a, b);
			(void)c;
		} catch (const std::invalid_argument&) {
			caught = true;
		}

		assert(caught);
	}

}  // namespace

int main() {
	TestGemmSmallFixed();
	TestGemmIdentity();
	TestGemmZeroMatrix();
	TestGemmShapeMismatch();

	std::cout << "All GEMM tests passed.\n";
	return 0;
}
