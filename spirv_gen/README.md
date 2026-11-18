# SYCL + SPIR-V vector add samples

This folder contains 3 minimal vector-add examples:

1. `1_opencl_to_spirv.cpp`
   - Compiles OpenCL C to SPIR-V via `oclocInvoke`.
2. `2_sycl_load_spirv.cpp`
   - Loads SPIR-V with SYCL (`create_kernel_bundle_from_source`) and runs it.
3. `3_opencl_to_spirv_to_sycl.cpp`
   - End-to-end path in one process: OpenCL C -> SPIR-V -> SYCL execution.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j
```

## Run

```bash
# 1) OpenCL C -> SPIR-V file
./build/sample1_opencl_to_spirv vector_add.spv

# 2) SYCL load SPIR-V and run
./build/sample2_sycl_load_spirv vector_add.spv

# 3) E2E in one binary
./build/sample3_opencl_to_spirv_to_sycl
```

## Notes

- Requires oneAPI SYCL runtime and `ocloc` development/runtime packages.
- If CMake cannot find `ocloc`, install package providing `ocloc_api.h` and `libocloc.so`.
