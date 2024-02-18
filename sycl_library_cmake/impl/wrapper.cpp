#include <sycl/sycl.hpp>
#include "vector_add.hpp"

void wrapper_vector_add(int* a, int* b, int* c, int N) {
    sycl::queue q_ct1{sycl::gpu_selector_v};
    // Call the kernel function on GPU using cudaMallocHost and cudaMemcpy functions
    int* a_dev = sycl::malloc_device<int>(N, q_ct1);
    int* b_dev = sycl::malloc_device<int>(N, q_ct1);
    int* c_dev = sycl::malloc_device<int>(N, q_ct1);

    q_ct1.memcpy(a_dev, a, sizeof(int) * N).wait();
    q_ct1.memcpy(b_dev, b, sizeof(int) * N).wait();

    vector_add(a_dev, b_dev, c_dev, N, q_ct1);

    q_ct1.memcpy(c, c_dev, sizeof(int) * N).wait();

    // Free device memory using cudaFree functions
    sycl::free(a_dev, q_ct1);
    sycl::free(b_dev, q_ct1);
    sycl::free(c_dev, q_ct1);
}
