#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

namespace
{

	void FillRandom(tinyinfer::Matrix &m, std::mt19937 &gen)
	{
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		for (std::size_t i = 0; i < m.rows(); ++i)
		{
			for (std::size_t j = 0; j < m.cols(); ++j)
			{
				m(i, j) = dist(gen);
			}
		}
	}

	void RunBenchmark(
		const tinyinfer::GemmBackend &gemm_backend,
		std::size_t m,
		std::size_t k,
		std::size_t n)
	{
		tinyinfer::Matrix a(m, k);
		tinyinfer::Matrix b(k, n);

		std::mt19937 gen(42);

		FillRandom(a, gen);
		FillRandom(b, gen);

		// warmup
		for (int i = 0; i < 3; ++i)
		{
			tinyinfer::Matrix c = gemm_backend.Compute(a, b);
			volatile float sink = c(0, 0);
			(void)sink;
		}

		std::size_t run_times = 3;
		std::vector<double> run_time(run_times);

		for (int i = 0; i < run_times; ++i)
		{

			const auto start = std::chrono::steady_clock::now();

			tinyinfer::Matrix c = gemm_backend.Compute(a, b);

			const auto end = std::chrono::steady_clock::now();

			volatile float sink = c(0, 0);
			(void)sink;

			const double ms =
				std::chrono::duration<double, std::milli>(end - start).count();

			run_time[i] = ms;
		}

		std::sort(run_time.begin(), run_time.end());
		double median_times = run_time[run_times / 2];
		
		const double flops = 2.0 * m * k * n;
		const double gflops = flops / (median_times / 1000.0) / 1e9;

		gemm_backend.PrintConfig(std::cout);
		std::cout << ", M = " << m
				  << ", K = " << k
				  << ", N = " << n
				  << ", time = " << median_times << " ms"
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
		tinyinfer::GemmIJKBackend gemmijk_backend;
		tinyinfer::GemmIKJBackend gemmikj_backend;
		RunBenchmark(gemmijk_backend, n, n, n);
		RunBenchmark(gemmikj_backend, n, n, n);

		for (const auto block_size : block_sizes)
		{
			tinyinfer::GemmBlockedBackend gemmblocked_backend(block_size);
			RunBenchmark(gemmblocked_backend, n, n, n);
		}

		std::cout << "\n";
	}

	return 0;
}
