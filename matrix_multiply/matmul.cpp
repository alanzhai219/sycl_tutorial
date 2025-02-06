#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::matrix;

constexpr int TM = 16; // Tile size M
constexpr int TN = 16; // Tile size N
constexpr int TK = 16; // Tile size K

void matmul_xmx(float* A, float* B, float* C, int M, int N, int K, sycl::nd_item<2> nit) {
    int row = nit.get_global_id(0);
    int col = nit.get_global_id(1);

    const auto sg_startx = row - nit.get_local_id(0);
    const auto sg_starty = col - nit.get_local_id(1);

    sycl::sub_group sg = nit.get_sub_group();

    auto pA = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(A);
    auto pB = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(B);
    auto pC = sycl::address_space_cast<
              sycl::access::address_space::global_space,
              sycl::access::decorated::no>(C);

    joint_matrix<sycl::sub_group, float, use::a, TM, TK, matrix::layout::row_major> subA;
    joint_matrix<sycl::sub_group, float, use::b, TK, TN, matrix::layout::col_major> subB;
    joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> subC;

    for (int k = 0; k < K; k += TK) {
        joint_matrix_load(sg, subA, pA + row * K + k, K);
        joint_matrix_load(sg, subB, pB + k * N + col, N);
        joint_matrix_mad(sg, subC, subA, subB, subC);
    }

    joint_matrix_store(sg, subC, pC + row * N + col, N, matrix::layout::row_major);
}

int main() {
    constexpr int M = 64, N = 64, K = 64;
    sycl::queue q(sycl::gpu_selector_v);

    sycl::device dev = q.get_device();
    std::cout << "## " << dev.get_info<sycl::info::device::name>() << "\n";

    float *A = malloc_shared<float>(M * K, q);
    float *B = malloc_shared<float>(K * N, q);
    float *C = malloc_shared<float>(M * N, q);

    std::fill_n(A, M * K, 1.0f);
    std::fill_n(B, K * N, 1.0f);
    std::fill_n(C, M * N, 0.0f);

    sycl::range<2> g_size(M, N);
    sycl::range<2> l_size(TM, TN);
    sycl::nd_range<2> nd_size(g_size, l_size);

    q.parallel_for(nd_size, [=](sycl::nd_item<2> nit) { matmul_xmx(A, B, C, M, N, K, nit); } );
    q.wait();

    std::cout << "Computation completed successfully." << std::endl;

    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}

