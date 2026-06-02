cmake_minimum_required(VERSION 3.16)

project(tiny_infer_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(tinyinfer
	src/core/matrix.cpp
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




