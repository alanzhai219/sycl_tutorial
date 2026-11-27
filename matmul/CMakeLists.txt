cmake_minimum_required(VERSION 3.20)
project(sycl-matmul-tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_CXX_COMPILER MATCHES "icx|icpx|dpcpp")
    message(WARNING "Use Intel oneAPI icpx for SYCL compilation")
endif()

add_executable(simd_gemm simd_gemm.cpp)
add_executable(esimd_gemm esimd_gemm.cpp)
add_executable(joint_matrix_gemm joint_matrix_gemm.cpp)

foreach(target simd_gemm esimd_gemm joint_matrix_gemm)
    target_compile_options(${target} PRIVATE -fsycl -O3 -Wall -Wextra)
    target_link_options(${target} PRIVATE -fsycl)
endforeach()
