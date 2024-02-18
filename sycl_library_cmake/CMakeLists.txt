cmake_minimum_required(VERSION 3.10)
project(add_vector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUORED ON)

add_subdirectory(impl)

add_executable(run
               main.cpp)

target_link_libraries(run
                      wrapper_vector_add)
