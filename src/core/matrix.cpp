#include "tinyinfer/core/matrix.h"

#include <algorithm>

namespace tinyinfer {

Matrix::Matrix()
    : rows_(0), cols_(0), data_() {}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, float value)
	:rows_(rows), cols_(cols), data_(rows * cols, value) {}

std::size_t Matrix::rows() const {
	return rows_;
}

std::size_t Matrix::cols() const {
	return cols_;
}

std::size_t Matrix::size() const {
	return data_.size();
}

float& Matrix::operator()(std::size_t row, std::size_t col) {
	return data_[index(row, col)];
}

const float& Matrix::operator()(std::size_t row, std::size_t col) const {
	return data_[index(row, col)];
}

const float* Matrix::data() const {
	return data_.data();
}

void Matrix::fill(float value) {
	std::fill(data_.begin(), data_.end(), value);
}

std::size_t Matrix::index(std::size_t row, std::size_t col) const {
	if (row >= rows_ || col >= cols_) {
		throw std::out_of_range("Matrix index out of range");
	}

	return row * cols_ + col;
}

} // namespace tinyinfer
