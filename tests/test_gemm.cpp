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

	void TestGemmIJKEqualsIKJ() {
		tinyinfer::Matrix a(3, 4);
		tinyinfer::Matrix b(4, 2);

		float value = 1.0f;

		for (std::size_t i = 0; i < a.rows(); ++i) {
			for (std::size_t j = 0; j < a.cols(); ++j) {
				a(i, j) = value;
				value += 1.0f;
			}
		}

		value = 1.0f;

		for (std::size_t i = 0; i < b.rows(); ++i) {
			for (std::size_t j = 0; j < b.cols(); ++j) {
				b(i, j) = value;
				value += 1.0f;
			}
		}

		tinyinfer::Matrix c_ijk = tinyinfer::GemmIJK(a, b);
		tinyinfer::Matrix c_ikj = tinyinfer::GemmIKJ(a, b);

		assert(c_ijk.rows() == c_ikj.rows());
		assert(c_ijk.cols() == c_ikj.cols());

		for (std::size_t i = 0;i < c_ijk.rows(); ++i) {
			for (std::size_t j = 0; j < c_ikj.cols(); ++j) {
				assert(c_ijk(i, j) == c_ikj(i, j));
			}
		}
	}

	void TestGemmBlockedEqualsIKJ() {
		tinyinfer::Matrix a(5, 7);
		tinyinfer::Matrix b(7, 4);

		float value = 1.0f;

		for (std::size_t i = 0; i < a.rows(); ++i) {
			for (std::size_t j = 0; j < a.cols(); ++j) {
				a(i, j) = value;
				value += 0.1f;
			}
		}

		value = -1.0f;

		for (std::size_t i = 0; i < b.rows(); ++i) {
			for (std::size_t j = 0; j < b.cols(); ++j) {
				b(i, j) = value;
				value += 0.2f;
			}
		}

		tinyinfer::Matrix expected = tinyinfer::GemmIKJ(a, b);
		
		const std::vector<std::size_t> block_sizes = {1,2,3,4,8};

		for (const auto block_size : block_sizes) {
			tinyinfer::Matrix actual = tinyinfer::GemmBlocked(a, b, block_size);

			assert(actual.rows() == expected.rows());
			assert(actual.cols() == expected.cols());

			for (std::size_t i = 0; i < actual.rows(); ++i) {
				for (std::size_t j = 0; j < actual.cols(); ++j) {
					assert(AlmostEqual(actual(i, j), expected(i, j), 1e-4));
				}
			}
		}
	}


}  // namespace

int main() {
	TestGemmSmallFixed();
	TestGemmIdentity();
	TestGemmZeroMatrix();
	TestGemmShapeMismatch();
	TestGemmIJKEqualsIKJ();
	TestGemmBlockedEqualsIKJ();

	std::cout << "All GEMM tests passed.\n";
	return 0;
}
