#include <sycl/sycl.hpp>
#include "vector_add.hpp"

void vector_add_kernel(int *a, int *b, int *c, int N,
                       const sycl::nd_item<3> &item_ct1) {
  // Get the global thread index
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  // Check if we are within bounds
  if (i < N) {
    // Add a[i] + b[i] to c[i]
    c[i] = a[i] + b[i];
  }
}

void vector_add(int* a, int* b, int* c, int N, sycl::queue& q) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    q.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
            vector_add_kernel(a, b, c, N, item_ct1);
        }).wait_and_throw();
}
