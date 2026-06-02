#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

namespace tinyinfer {

class Matrix {
public:
	Matrix();
	Matrix(std::size_t rows, std::size_t cols);
	Matrix(std::size_t rows, std::size_t cols, float value);

	std::size_t rows() const;
	std::size_t cols() const;
	std::size_t size() const;

	float& operator()(std::size_t row, std::size_t col);
	const float& operator()(std::size_t row, std::size_t col) const;

	float* data();
	const float* data() const;

	void fill(float value);

private:
	std::size_t rows_;
	std::size_t cols_;
	std::vector<float> data_;

	std::size_t index(std::size_t row, std::size_t col) const;
};
} // namespace tinyinfer

