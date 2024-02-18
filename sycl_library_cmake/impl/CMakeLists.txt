project(wrapper_vector_add)

find_program(ICPX_COMPILER icpx)

message("##### find the icpx compiler: ${ICPX_COMPILER}")
set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
# add_compile_options(-fsycl)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

add_library(wrapper_vector_add SHARED
            wrapper.cpp
            vector_add.cpp)

