#include <sycl/sycl.hpp>

void vector_add(float* a,
                float* b,
                float* c,
                int n,
                sycl::nd_item<3>& ndi) {
    size_t work_item_idx = ndi.get_local_id(2); 
    size_t work_group_idx = ndi.get_group(2);
    size_t work_group_size = ndi.get_local_range().get(2);
    size_t idx = work_item_idx + work_group_idx * work_group_size;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void init_memory(float* ptr, size_t N) {
    for (size_t i=0; i < N; ++i) {
        ptr[i] = i;
    }
}

int main() {
    const size_t N = 256;
    const size_t sub_N = 16;

    // 1. create a queue
    sycl::queue q(sycl::gpu_selector_v);

    // 2. malloc memory
    // 2.1. malloc host memory
    float* a_host = static_cast<float*>(sycl::malloc_host(sizeof(float) * N, q));
    float* b_host = static_cast<float*>(sycl::malloc_host(sizeof(float) * N, q));
    float* c_host = static_cast<float*>(sycl::malloc_host(sizeof(float) * N, q));

    // 2.2. init memory
    init_memory(a_host, N);
    init_memory(b_host, N);
    q.memset(c_host, 0, sizeof(float) * N);

    // 2.3. malloc device memory
    float* a_dev = static_cast<float*>(sycl::malloc_device(sizeof(float) * N, q));
    float* b_dev = static_cast<float*>(sycl::malloc_device(sizeof(float) * N, q));
    float* c_dev = static_cast<float*>(sycl::malloc_device(sizeof(float) * N, q));

    // 3. copy host memory to device memory
    q.memcpy(a_dev, a_host, sizeof(float) * N);
    q.memcpy(b_dev, b_host, sizeof(float) * N);

    // 4. launch kernel
    sycl::range glb_size{1, 1, N};
    sycl::range loc_size{1, 1, sub_N};
    sycl::nd_range parallel_size{glb_size, loc_size};
    q.parallel_for(parallel_size, [=](sycl::nd_item<3> nit) { vector_add(a_dev, b_dev, c_dev, N, nit); });
    q.wait();

    // 5. copy device memory to host memory
    q.memcpy(c_host, c_dev, sizeof(float) * N);

    // 6. print
    for (size_t i = 0; i < 10; ++i) {
        std::cout << c_host[i] << "\n";
    }

    // 7. free
    sycl::free(a_host, q);
    sycl::free(b_host, q);
    sycl::free(c_host, q);

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(c_dev, q);

    return 0;
}
