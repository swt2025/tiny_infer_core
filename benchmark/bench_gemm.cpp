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

	void RunBenchmark(std::size_t n) {
		tinyinfer::Matrix a(n, n);
		tinyinfer::Matrix b(n, n);

		FillRandom(a);
		FillRandom(b);

		// warmup
		volatile float sink = 0.0f;
		{
			tinyinfer::Matrix c = tinyinfer::GemmNaive(a, b);
			sink = c(0, 0);
		}

		const auto start = std::chrono::high_resolution_clock::now();

		tinyinfer::Matrix c = tinyinfer::GemmNaive(a, b);

		const auto end = std::chrono::high_resolution_clock::now();

		const double ms =
			std::chrono::duration<double, std::milli>(end - start).count();

		const double flops = 2.0 * n * n * n;
		const double gflops = flops / (ms / 1000.0) / 1e9;

		std::cout << "N = " << n
			<< ", time = " << ms << " ms"
			<< ", GFLOPS = " << gflops
			<< ", sink = " << sink
			<< "\n";
	}

}  // namespace

int main() {
	const std::vector<std::size_t> sizes = {64, 128, 256, 512};

	for (const auto n : sizes) {
		RunBenchmark(n);
	}

	return 0;
}
