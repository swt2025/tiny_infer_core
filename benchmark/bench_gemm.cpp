#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace
{

	void FillRandom(tinyinfer::Matrix &m)
	{
		std::mt19937 gen(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		for (std::size_t i = 0; i < m.rows(); ++i)
		{
			for (std::size_t j = 0; j < m.cols(); ++j)
			{
				m(i, j) = dist(gen);
			}
		}
	}

	using GemmFunc = tinyinfer::Matrix (*)(const tinyinfer::Matrix &,
										   const tinyinfer::Matrix &);

	void RunBenchmark(
		const tinyinfer::GemmBackend &gemm_backend,
		std::size_t m,
		std::size_t k,
		std::size_t n)
	{
		tinyinfer::Matrix a(m, k);
		tinyinfer::Matrix b(k, n);

		FillRandom(a);
		FillRandom(b);

		// warmup
		{
			tinyinfer::Matrix c = gemm_backend.Compute(a, b);
			volatile float sink = c(0, 0);
			(void)sink;
		}

		const auto start = std::chrono::high_resolution_clock::now();

		tinyinfer::Matrix c = gemm_backend.Compute(a, b);

		const auto end = std::chrono::high_resolution_clock::now();

		volatile float sink = c(0, 0);
		(void)sink;

		const double ms =
			std::chrono::duration<double, std::milli>(end - start).count();

		const double flops = 2.0 * m * k * n;
		const double gflops = flops / (ms / 1000.0) / 1e9;

		gemm_backend.PrintConfig(std::cout);
		std::cout << ", M = " << m
				  << ", K = " << k
				  << ", N = " << n
				  << ", time = " << ms << " ms"
				  << ", GFLOPS = " << gflops
				  << "\n";
	}

} // namespace

int main()
{
	const std::vector<std::size_t> sizes = {128, 256, 512, 1024, 2048};
	const std::vector<std::size_t> block_sizes = {16, 32, 64, 128};

	for (const auto n : sizes)
	{
		const tinyinfer::GemmBackend &gemmijk_backend = tinyinfer::GemmIJKBackend();
		const tinyinfer::GemmBackend &gemmikj_backend = tinyinfer::GemmIKJBackend();
		RunBenchmark(gemmijk_backend, n, n, n);
		RunBenchmark(gemmikj_backend, n, n, n);

		for (const auto block_size : block_sizes)
		{
			const tinyinfer::GemmBackend &gemmblocked_backend = tinyinfer::GemmBlockedBackend(block_size);
			RunBenchmark(gemmblocked_backend, n, n, n);
		}

		std::cout << "\n";
	}

	return 0;
}
