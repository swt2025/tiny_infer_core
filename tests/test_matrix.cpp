#include "tinyinfer/core/matrix.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using tinyinfer::Matrix;

static bool almost_equal(float a, float b, float eps = 1e-6f) {
	return std::abs(a - b) < eps;
}

void test_create_matrix() {
	Matrix m(2, 3);

	assert(m.rows() == 2);
	assert(m.cols() == 3);
	assert(m.size() == 6);

	for (std::size_t i = 0; i < m.rows(); ++i) {
		for (std::size_t j = 0; j < m.cols(); ++j) {
			assert(almost_equal(m(i,j), 0.0f));
		}
	}
}

void test_init_value() {
	Matrix m(2, 2, 3.5f);
	assert(almost_equal(m(0,0), 3.5f));
	assert(almost_equal(m(0, 1), 3.5f));
	assert(almost_equal(m(1, 0), 3.5f));
	assert(almost_equal(m(1, 1), 3.5f));
}

void test_set_and_get() {
	Matrix m(2,3);
	m(0,0) = 1.0f;
	m(0,1) = 2.0f;
	m(1,2) = 6.0f;

	assert(almost_equal(m(0, 0), 1.0f));
	assert(almost_equal(m(0, 1), 2.0f));
	assert(almost_equal(m(1, 2), 6.0f));
}

void test_fill() {
	Matrix m(3,3);
	m.fill(7.0f);

	for(std::size_t i = 0; i < m.rows(); ++i) {
		for (std::size_t j = 0; j < m.cols(); ++j) {
			assert(almost_equal(m(i,j), 7.0f));
		}
	}
}


void test_out_of_range() {
	Matrix m(2, 2);

	bool caught = false;

	try {
		m(2,0) = 1.0f;
	} catch (const std::out_of_range&) {
		caught = true;
	}
	assert(caught);
}

int main() {
	test_create_matrix();
	test_init_value();
	test_set_and_get();
	test_fill();
	test_out_of_range();

	std::cout << "All Matrix tests passed.\n";
	return 0;
}
