# joint_matrix capability query

Print the joint_matrix combinations reported by each visible SYCL GPU. Each entry includes A/B input types, C accumulator type, D output type, and M/N/K tile dimensions for `D = A * B + C`.

## Build and run

```sh
source /opt/intel/oneapi/setvars.sh --force
icpx -std=c++17 -fsycl -O2 -Wall -Wextra main.cpp -o query_xmx
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./query_xmx
```

Run without the selector to query all visible GPU backends. Level Zero and OpenCL may expose the same physical GPU; the platform and driver are printed to distinguish them.

## Output

```text
input dtype: A=INT8, B=INT8
accumulator dtype: C=INT32
output dtype: D=INT32
shape: M=[1..8], N=16, K=32
```

`[1..8]` indicates a dimension range reported by the runtime. A fixed number indicates an exact dimension. These dimensions apply to joint_matrix tiles, not the full input matrices. Layout and subgroup requirements must also be satisfied.

The query does not execute matrix operations, verify generated XMX instructions, or measure performance. Results depend on the selected device, compiler, and runtime.