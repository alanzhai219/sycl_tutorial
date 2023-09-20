# Select between the USM and buffer variant of the
# code to compile, depending on the value of USM 
# given to cmake
# e.g. if cmake is called with -DUSM=1, the USM
# source code will be compiled
if(DEFINED USM AND (NOT(USM EQUAL 0)))
    message(STATUS "Using the USM variant.")
    set(SOURCE_FILE vector-add-usm.cpp)
    set(TARGET_NAME vector-add-usm)
else()    
    set(SOURCE_FILE vector-add-buffers.cpp)
    set(TARGET_NAME vector-add-buffers)
endif()

# 
# SECTION 1
# This section defines rules to create a cpu-gpu make target
# This can safely be removed if your project is only targetting FPGAs
#

set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")

# To compile in a single command:
#    icpx -fsycl <file>.cpp -o <file>.fpga_emu
# CMake executes:
#    [compile] icpx -fsycl -o <file>.cpp.o -c <file>.cpp
#    [link]    icpx -fsycl <file>.cpp.o -o <file>.fpga_emu
add_executable(${TARGET_NAME} ${SOURCE_FILE})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_custom_target(cpu-gpu DEPENDS ${TARGET_NAME})

# 
# End of SECTION 1
#
