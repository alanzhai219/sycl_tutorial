cmake_minimum_required(VERSION 3.20)
project(sycl_spirv_samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(IntelSYCL REQUIRED)

set(SYCL_LINK_LIB "")
if(TARGET IntelSYCL::SYCL_CXX)
	set(SYCL_LINK_LIB IntelSYCL::SYCL_CXX)
elseif(TARGET IntelSYCL::SYCL)
	set(SYCL_LINK_LIB IntelSYCL::SYCL)
else()
	set(SYCL_LINK_LIB sycl)
endif()

add_executable(sample1_opencl_to_spirv 1_opencl_to_spirv.cpp)
target_link_libraries(sample1_opencl_to_spirv PRIVATE ${SYCL_LINK_LIB} ocloc)

add_executable(sample2_sycl_load_spirv 2_sycl_load_spirv.cpp)
target_link_libraries(sample2_sycl_load_spirv PRIVATE ${SYCL_LINK_LIB})

add_executable(sample3_opencl_to_spirv_to_sycl 3_opencl_to_spirv_to_sycl.cpp)
target_link_libraries(sample3_opencl_to_spirv_to_sycl PRIVATE ${SYCL_LINK_LIB} ocloc)
