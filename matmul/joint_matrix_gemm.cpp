#include "common.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

int main(int argc, char **argv) {
    try {
        const options opt = parse_options(argc, argv);
        if ((opt.m % 8) != 0 || (opt.n % 16) != 0 || (opt.k % 16) != 0) {
            throw std::invalid_argument("joint_matrix FP16 requires M multiple of 8, N multiple of 16, and K multiple of 16");
        }

        sycl::queue queue = make_queue();
        const std::size_t count_a = opt.m * opt.k;
        const std::size_t count_b = opt.k * opt.n;
        const std::size_t count_c = opt.m * opt.n;

        std::vector<sycl::half> ha(count_a), hb(count_b);
        std::vector<float> hc(count_c);
        for (std::size_t i = 0; i < count_a; ++i) ha[i] = static_cast<sycl::half>(random_float(i, 17));
        for (std::size_t i = 0; i < count_b; ++i) hb[i] = static_cast<sycl::half>(random_float(i, 101));

        sycl::half *a = sycl::malloc_device<sycl::half>(count_a, queue);
        sycl::half *b = sycl::malloc_device<sycl::half>(count_b, queue);
        float *c = sycl::malloc_device<float>(count_c, queue);
        if (!a || !b || !c) throw std::runtime_error("device allocation failed");

        queue.memcpy(a, ha.data(), count_a * sizeof(sycl::half)).wait_and_throw();
        queue.memcpy(b, hb.data(), count_b * sizeof(sycl::half)).wait_and_throw();

        constexpr std::size_t m_tile = 8;
        constexpr std::size_t n_tile = 16;
        constexpr std::size_t k_tile = 16;
        constexpr std::size_t subgroup_size = 16;
        const std::size_t tile_count = (opt.m / m_tile) * (opt.n / n_tile);

        auto submit = [&]() {
            return queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(tile_count * subgroup_size),
                                                       sycl::range<1>(subgroup_size)),
                    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                        using namespace sycl::ext::oneapi::experimental::matrix;
                        using a_matrix = joint_matrix<sycl::sub_group, sycl::half, use::a,
                                                      m_tile, k_tile, layout::row_major>;
                        using b_matrix = joint_matrix<sycl::sub_group, sycl::half, use::b,
                                                      k_tile, n_tile, layout::row_major>;
                        using c_matrix = joint_matrix<sycl::sub_group, float, use::accumulator,
                                                      m_tile, n_tile, layout::dynamic>;

                        const std::size_t tile = item.get_group(0);
                        const std::size_t n_tiles = opt.n / n_tile;
                        const std::size_t tile_row = tile / n_tiles;
                        const std::size_t tile_col = tile % n_tiles;
                        const std::size_t row = tile_row * m_tile;
                        const std::size_t col = tile_col * n_tile;
                        sycl::sub_group sg = item.get_sub_group();

                        c_matrix result;
                        c_matrix zero;
                        joint_matrix_fill(sg, result, 0.0f);
                        joint_matrix_fill(sg, zero, 0.0f);

                        a_matrix am;
                        b_matrix bm;
                        for (std::size_t inner = 0; inner < opt.k; inner += k_tile) {
                            joint_matrix_load(sg, am,
                                sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(a + row * opt.k + inner),
                                opt.k);
                            joint_matrix_load(sg, bm,
                                sycl::address_space_cast<sycl::access::address_space::global_space,
                                                         sycl::access::decorated::no>(b + inner * opt.n + col),
                                opt.n);
                            if (inner == 0) {
                                joint_matrix_mad(sg, result, am, bm, zero);
                            } else {
                                joint_matrix_mad(sg, result, am, bm, result);
                            }
                        }

                        joint_matrix_store(sg, result,
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(c + row * opt.n + col),
                            opt.n, layout::row_major);
                    });
            });
        };

        for (int i = 0; i < 5; ++i) submit().wait_and_throw();
        std::vector<double> times;
        for (int i = 0; i < opt.runs; ++i) {
            sycl::event event = submit();
            event.wait_and_throw();
            const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
            times.push_back((end - start) * 1e-9);
        }

        queue.memcpy(hc.data(), c, count_c * sizeof(float)).wait_and_throw();
        const std::vector<float> reference_a(ha.begin(), ha.end());
        const std::vector<float> reference_b(hb.begin(), hb.end());
        validate(hc.data(), make_reference(reference_a, reference_b, opt.m, opt.n, opt.k), count_c);

        const double seconds = median(times);
        const double gflops = 2.0 * opt.m * opt.n * opt.k / seconds / 1e9;
        std::cout << std::fixed << std::setprecision(3)
                  << "Joint-matrix GEMM (FP16 input, FP32 accumulator/output): " << opt.m << "x" << opt.k << " * "
                  << opt.k << "x" << opt.n << ", median=" << seconds * 1000
                  << " ms, " << gflops << " GFLOPS, PASS\n";

        sycl::free(a, queue);
        sycl::free(b, queue);
        sycl::free(c, queue);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
