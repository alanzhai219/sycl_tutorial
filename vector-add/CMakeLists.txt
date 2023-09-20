cmake_minimum_required (VERSION 3.4)

# Direct CMake to use icpx rather than the default C++ compiler/linker
# icpx is DPC++ compiler while dpcpp is deprecated.
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(VectorAdd CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (src)
