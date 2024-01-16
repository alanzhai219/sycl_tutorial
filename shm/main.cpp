#include <sycl/sycl.hpp>

constexpr size_t N = 1024;

int main() {
    // 创建队列
    sycl::queue q;

    // 分配设备端内存
    int* data_h = sycl::malloc_host<int>(N, q);
    for (int i=0; i<N; ++i) {
        data_h[i] = i;
    }
    int* data_d = sycl::malloc_host<int>(N, q);
    q.memcpy(data_d, data_h, sizeof(int) * N).wait();

    // 使用 nd_range
    sycl::range<1> globalSize(N);
    sycl::range<1> localSize(64); // 假设每个工作组有 64 个工作项
    sycl::nd_range<1> ndRange(globalSize, localSize);

    // 启动内核
    q.parallel_for(ndRange, [=](sycl::nd_item<1> item) {
        auto local_mem = sycl::ext::oneapi::group_local_memory<int[64]>(item.get_group());
        // 计算全局 ID
        size_t globalId = item.get_global_id(0);
        size_t localId = item.get_local_id(0);

        // 在 USM 中进行计算，这里只是一个示例，您可以根据需要进行修改
        data_d[globalId] = localId * 2;
    }).wait();

    // 在主机上验证结果
    q.memcpy(data_h, data_d, N * sizeof(int)).wait();

    // 打印结果
    for (size_t i = 0; i < N; ++i) {
        std::cout << data_h[i] << " ";
    }

    // 释放 USM 内存
    sycl::free(data_h, q);
    sycl::free(data_d, q);

    return 0;
}

