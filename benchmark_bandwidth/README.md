# SYCL GPU 内存带宽微基准

用于比较 dGPU 访问 GDDR 与 iGPU 访问系统 DDR 的持续有效带宽。不依赖 llama.cpp，也不测 CPU/GPU 之间的 PCIe memcpy。

## 构建

在本目录运行，要求 Intel oneAPI DPC++/SYCL 编译器和可用 GPU 驱动：

```bash
source /opt/intel/oneapi/setvars.sh --force
icpx -O3 -std=c++17 -fsycl main.cpp -o bandwidth
```

## 运行

```bash
./bandwidth --help
./bandwidth --list
./bandwidth --device 0 --mib 256 --iterations 50 --warmup 5 --local 256
```

`--device` 使用本程序 `--list` 的编号。平台不同可能重复枚举同一物理 GPU；比较时优先选择相同运行时，例如都使用 Level Zero，而不是把 OpenCL/Level Zero 实例当成两张显卡。

`--mib` 是每个数组的大小，共分配三个独立 FP32 device USM 数组。例如 `--mib 512` 共使用 1.5 GiB 设备内存。程序检查单次分配限制，并要求三个数组不超过报告设备内存的 75%；实际可用内存可能更少，分配失败时应减小规模。

如果列出了两张不同物理 GPU，应串行运行：

```bash
./bandwidth --device 0 --mib 512 --iterations 100
./bandwidth --device 1 --mib 512 --iterations 100
```

建议按每数组 32、128、512 MiB 扫描规模。选择明显超过相关缓存容量且没有内存压力的稳定平台区，不要把小规模缓存带宽当成 GDDR/DDR 带宽。

## 操作和计量

| 操作 | 表达式 | FP32 每元素逻辑流量 |
| --- | --- | ---: |
| Copy | `a[i] = b[i]` | 8 bytes，1 读 1 写 |
| Add | `a[i] = b[i] + c[i]` | 12 bytes，2 读 1 写 |
| Triad | `a[i] = b[i] + 2*c[i]` | 12 bytes，2 读 1 写 |

三个数组先由 GPU 初始化。每个操作分别预热，然后逐次执行并等待，使用 GPU event 的 `command_end - command_start` 计时。分配、初始化、CPU 提交、预热和结果回读不包含在 kernel 时间内。输出中位延迟，以及按中位、最快、最慢延迟计算的有效带宽。

计算公式：有效 GB/s = 元素数 x 每元素逻辑 bytes / GPU 时间秒 / 1e9。

每个操作结束后，分块回读并检查全部输出，失败则退出码非零。输入使用可精确表示的二进制分数，避免把浮点舍入差异当成带宽测试错误。测试不会在计时期间把大数组搬回主机。

## 结果的边界

- 这是算法有效带宽，不是硬件计数器实测 DRAM bytes。缓存、写入策略和物理事务粒度会让两者不同。
- device USM 表达设备分配意图，但具体物理驻留和驱动策略由平台决定；不要仅凭 API 声称已验证 GDDR 驻留。
- iGPU 的 DDR 与 CPU、显示等共享。关闭其他 GPU 任务及 CPU 内存压力，并保持电源和散热条件一致。
- 每个 kernel 每个元素对应一个 work-item；它是可读的基线，不保证达到每张 GPU 的最高带宽。可比较不同 `--local`，并用成熟的 BabelStream 交叉验证。
- 设备列表中的 device/shared USM 支持信息不能识别物理内存类型。GDDR/DDR 型号、速率和总位宽需另外确认。
- 需要实际 DRAM 流量时，应进一步采集平台支持的 VTune GPU 内存硬件指标。报告中的 Host-to-Device Transfer Size 不能代替 kernel 的内存流量。
- 不应直接用这里的带宽比值预测 dequant/GEMM 加速比；那些算子还有解码、计算和同步成本。

理论峰值可按有效数据率 x 总位宽 / 8 计算；DDR 的 MT/s 已经是有效传输率，不要再乘二。
