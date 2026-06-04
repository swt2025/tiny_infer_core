cmake_minimum_required(VERSION 3.16)

project(tiny_infer_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tinyinfer
	src/core/matrix.cpp
	src/ops/gemm.cpp
	src/ops/softmax.cpp
)

target_include_directories(tinyinfer
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(test_matrix
	tests/test_matrix.cpp
)

target_link_libraries(test_matrix
	PRIVATE
	tinyinfer
)

add_executable(test_gemm
	tests/test_gemm.cpp
)

target_link_libraries(test_gemm
	PRIVATE
	tinyinfer
)

add_executable(test_softmax
	tests/test_softmax.cpp
)
target_link_libraries(test_softmax
	PRIVATE
	tinyinfer
)

add_executable(bench_gemm
    benchmark/bench_gemm.cpp
)

target_link_libraries(bench_gemm
    PRIVATE
        tinyinfer
)

