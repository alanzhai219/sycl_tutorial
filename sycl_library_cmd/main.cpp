#include <iostream>
#include <sycl/sycl.hpp>
#include "vector_add.hpp"

int main() {
    const int N = 16;
    // Allocate device memory for two vectors of size 16
    int* a = static_cast<int*>(malloc(sizeof(int) * N));
    int* b = static_cast<int*>(malloc(sizeof(int) * N));
    int* c = static_cast<int*>(malloc(sizeof(int) * N));

    // Initialize the vectors with random values
    for (int i=0; i<N; i++) {
        a[i]=rand()%100;
        b[i]=rand()%100;
        c[i]=rand()%100;
    }

    sycl::queue q_ct1{sycl::gpu_selector_v};
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

    // Print the results using cudaGetLastError function
    std::cout<<"All done"<<std::endl;
    for (int i=0; i<N; i++) {
        printf("a[%d] + b[%d] = %d\n", a[i], b[i], c[i]);
    }

    //
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
