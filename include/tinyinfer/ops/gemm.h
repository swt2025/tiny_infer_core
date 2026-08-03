#pragma once
#include <ostream>
#include <cstddef>
#include <stdexcept>
#include <string_view>

#include "tinyinfer/core/matrix.h"

namespace tinyinfer
{

	Matrix GemmNaive(const Matrix &a, const Matrix &b);
	Matrix GemmIJK(const Matrix &a, const Matrix &b);
	Matrix GemmIKJ(const Matrix &a, const Matrix &b);

	Matrix GemmBlocked(const Matrix &a, const Matrix &b, std::size_t block_size);

	enum class GemmType
	{
		Naive,
		IJK,
		IKJ,
		Blocked
	};

	class GemmBackend
	{
	public:
		virtual ~GemmBackend() = default;

		virtual Matrix Compute(const Matrix &a,
							   const Matrix &b) const = 0;

		virtual std::string_view Name() const noexcept = 0;

		virtual void PrintConfig(std::ostream &output) const
		{
			output << Name();
		}
	};

	class GemmIJKBackend final : public GemmBackend
	{
	public:
		Matrix Compute(const Matrix &a,
					   const Matrix &b) const override
		{
			return GemmIJK(a, b);
		}

		std::string_view Name() const noexcept override
		{
			return "GemmIJK";
		}
	};

	class GemmIKJBackend final : public GemmBackend
	{
	public:
		Matrix Compute(const Matrix &a,
					   const Matrix &b) const override
		{
			return GemmIKJ(a, b);
		}

		std::string_view Name() const noexcept override
		{
			return "GemmIKJ";
		}
	};

	class GemmBlockedBackend final : public GemmBackend
	{
	public:
		explicit GemmBlockedBackend(std::size_t block_size) : block_size_(block_size)
		{
			if (block_size == 0)
			{
				throw std::invalid_argument(
					"Blocked size must be greater than zero.");
			}
		};

		Matrix Compute(const Matrix &a,
					   const Matrix &b) const override
		{
			return GemmBlocked(a, b, block_size_);
		}
		std::string_view Name() const noexcept override
		{
			return "GemmBlocked";
		}

		void PrintConfig(std::ostream &output) const override
		{
			output << Name() << " block_size: " << block_size_;
		}

	private:
		std::size_t block_size_;
	};
} // namespace tinyinfer
