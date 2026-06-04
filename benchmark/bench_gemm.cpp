#include "tinyinfer/core/matrix.h"
#include "tinyinfer/ops/gemm.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

namespace {

	void FillRandom(tinyinfer::Matrix& m) {
		std::mt19937 gen(42);
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		for (std::size_t i = 0; i < m.rows(); ++i) {
			for (std::size_t j = 0; j < m.cols(); ++j) {
				m(i, j) = dist(gen);
			}
		}
	}

	using GemmFunc = tinyinfer::Matrix(*)(const tinyinfer::Matrix&,
			const tinyinfer::Matrix&);

	void RunBenchmark(const std::string& name, GemmFunc gemm_func, std::size_t n) {
		tinyinfer::Matrix a(n, n);
		tinyinfer::Matrix b(n, n);

		FillRandom(a);
		FillRandom(b);

		// warmup
		{
			tinyinfer::Matrix c = gemm_func(a, b);
			volatile float sink = c(0, 0);
			(void)sink;
		}

		const auto start = std::chrono::high_resolution_clock::now();

		tinyinfer::Matrix c = gemm_func(a, b);

		const auto end = std::chrono::high_resolution_clock::now();

		volatile float sink = c(0, 0);
		(void)sink;

		const double ms =
			std::chrono::duration<double, std::milli>(end - start).count();

		const double flops = 2.0 * n * n * n;
		const double gflops = flops / (ms / 1000.0) / 1e9;

		std::cout << name 
			<< ", N = " << n
			<< ", time = " << ms << " ms"
			<< ", GFLOPS = " << gflops
			<< ", sink = " << sink
			<< "\n";
	}

	void RunBlockedBenchmark(std::size_t n, std::size_t block_size) {
		tinyinfer::Matrix a(n, n);
		tinyinfer::Matrix b(n, n);

		FillRandom(a);
		FillRandom(b);

		{
			tinyinfer::Matrix warmup = tinyinfer::GemmBlocked(a, b, block_size);
			volatile float sink = warmup(0,0);
			(void)sink;
		}

		const auto start = std::chrono::high_resolution_clock::now();

		tinyinfer::Matrix c = tinyinfer::GemmBlocked(a, b, block_size);

		const auto end = std::chrono::high_resolution_clock::now();

		volatile float sink = c(0, 0);
		(void)sink;

		const double ms = std::chrono::duration<double, std::milli>(end - start).count();

		const double flops = 2 * n * n * n;
		const double gflops = flops / (ms / 1000.0) / 1e9;

		std::cout << "GemmBlocked"
			<< ", N = " << n
			<< ", block_size = " << block_size
			<< ", time = " << ms << " ms"
			<< ", GFLOPS = " << gflops
			<< "\n";
	}
}  // namespace

int main() {
	const std::vector<std::size_t> sizes = {128, 256, 512, 1024};
	const std::vector<std::size_t> block_sizes = {16, 32, 64, 128};

	for (const auto n : sizes) {
		RunBenchmark("GemmIJK",tinyinfer::GemmIJK, n);
		RunBenchmark("GemmIKJ",tinyinfer::GemmIKJ, n);

		for (const auto block_size : block_sizes) {
			RunBlockedBenchmark(n, block_size);
		}

		std::cout << "\n";
	}

	return 0;
}
