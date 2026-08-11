cmake_minimum_required(VERSION 3.16)

project(tiny_infer_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tinyinfer
	src/core/matrix.cpp
	src/ops/gemm.cpp
	src/ops/softmax.cpp
	src/ops/rms_norm.cpp
	src/ops/silu.cpp
	src/ops/swiglu.cpp
	src/ops/linear.cpp
	src/layers/mlp.cpp
	src/ops/rope.cpp
	src/ops/transpose.cpp
	src/ops/causal_mask.cpp
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

add_executable(test_rms_norm
	tests/test_rms_norm.cpp
)

target_link_libraries(test_rms_norm
	PRIVATE
	tinyinfer
)

add_executable(test_silu
	tests/test_silu.cpp
)

target_link_libraries(test_silu
	PRIVATE
	tinyinfer
)

add_executable(test_swiglu
	tests/test_swiglu.cpp
) 

target_link_libraries(test_swiglu
	PRIVATE
	tinyinfer
)

add_executable(test_linear
	tests/test_linear.cpp
)

target_link_libraries(test_linear
	PRIVATE
	tinyinfer
)

add_executable(test_mlp
	tests/test_mlp.cpp
)

target_link_libraries(test_mlp
	PRIVATE
	tinyinfer
)


add_executable(test_rope
	tests/test_rope.cpp
)

target_link_libraries(test_rope
	PRIVATE
	tinyinfer
)


add_executable(test_transpose
	tests/test_transpose.cpp
)

target_link_libraries(test_transpose
	PRIVATE
	tinyinfer
)

add_executable(
    test_causal_mask
    tests/test_causal_mask.cpp
)

target_link_libraries(
    test_causal_mask
    PRIVATE tinyinfer
)

enable_testing()

add_test(NAME test_matrix COMMAND test_matrix)
add_test(NAME test_gemm COMMAND test_gemm)
add_test(NAME test_softmax COMMAND test_softmax)
add_test(NAME test_rms_norm COMMAND test_rms_norm)
add_test(NAME test_silu COMMAND test_silu)
add_test(NAME test_swiglu COMMAND test_swiglu)
add_test(NAME test_linear COMMAND test_linear)
add_test(NAME test_mlp COMMAND test_mlp)
add_test(NAME test_rope COMMAND test_rope)
add_test(NAME test_transpose COMMAND test_transpose)


add_test(
    NAME test_causal_mask
    COMMAND test_causal_mask
)