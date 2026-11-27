# 三种 SYCL GEMM 实现

本目录提供三份独立实现：

- `simd_gemm.cpp`：普通 SYCL，每个 work-item 计算一个输出元素。
- `esimd_gemm.cpp`：ESIMD，一次显式处理 16列输出。
- `joint_matrix_gemm.cpp`：SYCL `joint_matrix`，FP16 输入、FP32 累加和输出，tile 为 8x16x16。

普通和 ESIMD 使用 FP32，joint_matrix 使用 FP16 输入。这不是完全相同精度的性能比较。

## 编译

```bash
source /opt/intel/oneapi/setvars.sh --force
mkdir -p build

icpx -std=c++20 -fsycl -O3 simd_gemm.cpp -o build/simd_gemm
icpx -std=c++20 -fsycl -O3 esimd_gemm.cpp -o build/esimd_gemm
icpx -std=c++20 -fsycl -O3 joint_matrix_gemm.cpp -o build/joint_matrix_gemm
```

## 运行

```bash
ONEAPI_DEVICE_SELECTOR='level_zero:gpu' ./build/simd_gemm 256 256 256 10
ONEAPI_DEVICE_SELECTOR='level_zero:gpu' ./build/esimd_gemm 256 256 256 10
ONEAPI_DEVICE_SELECTOR='level_zero:gpu' ./build/joint_matrix_gemm 256 256 256 10
```

参数顺序：`M N K [RUNS]`。

joint_matrix要求：

- M 是 8 的倍数。
- N、K 是 16 的倍数。
- A、B 输入是 FP16。
- 累加器和输出是 FP32，subgroup size 为 16。

## 正确性

每份实现会在GPU计算后回读结果，并与主机参考结果比较：

- 所有输出使用 FP32，误差上限为 `1e-4 * max(1, abs(reference))`。
- joint_matrix 的主机参考结果基于实际舍入后的 FP16 输入，以 double 计算点积，再转换为 FP32。

## 当前 Arc B390观察

当前运行时报告支持 FP16 输入、FP32 累加和输出的 `M=[1..8], N=16, K=16` 组合。本实现使用 8x16x16；此前的 N=8 tile 不受支持，不能据此认定 B390 不支持 joint_matrix。

查询报告与运行校验不等于 GPU ISA 证据。确认 XMX 指令使用时，可检查：

- IGC编译日志或GPU ISA中出现DPAS/DPASW。
- VTune显示对应matrix/DPAS活动。
- 普通SIMD、ESIMD和joint_matrix形状、精度一致且相互比较。
