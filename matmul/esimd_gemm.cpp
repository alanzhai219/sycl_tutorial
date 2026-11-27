#include "common.hpp"

#include <sycl/ext/intel/esimd.hpp>

int main(int argc, char **argv) {
    try {
        const options opt = parse_options(argc, argv);
        sycl::queue queue = make_queue();

        const std::size_t count_a = opt.m * opt.k;
        const std::size_t count_b = opt.k * opt.n;
        const std::size_t count_c = opt.m * opt.n;
        std::vector<float> ha(count_a), hb(count_b), hc(count_c);
        for (std::size_t i = 0; i < count_a; ++i) ha[i] = random_float(i, 17);
        for (std::size_t i = 0; i < count_b; ++i) hb[i] = random_float(i, 101);

        float *a = sycl::malloc_device<float>(count_a, queue);
        float *b = sycl::malloc_device<float>(count_b, queue);
        float *c = sycl::malloc_device<float>(count_c, queue);
        if (!a || !b || !c) throw std::runtime_error("device allocation failed");

        queue.memcpy(a, ha.data(), count_a * sizeof(float)).wait_and_throw();
        queue.memcpy(b, hb.data(), count_b * sizeof(float)).wait_and_throw();

        constexpr int width = 16;
        const std::size_t column_tiles = (opt.n + width - 1) / width;
        const std::size_t groups = opt.m * column_tiles;
        const std::size_t workgroup = 16;

        auto submit = [&]() {
            auto kernel = [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const std::size_t work = item.get_global_id(0);
                if (work >= groups) return;

                const std::size_t row = work / column_tiles;
                const std::size_t col = (work % column_tiles) * width;

                sycl::ext::intel::esimd::simd<float, width> result(0.0f);
                for (std::size_t inner = 0; inner < opt.k; ++inner) {
                    const float av = a[row * opt.k + inner];
                    if (col + width <= opt.n) {
                        sycl::ext::intel::esimd::simd<float, width> bv;
                        bv.copy_from(b + inner * opt.n + col,
                                     sycl::ext::intel::esimd::element_aligned_tag{});
                        result += av * bv;
                    } else {
                        for (int lane = 0; lane < static_cast<int>(opt.n - col); ++lane) {
                            result[lane] += av * b[inner * opt.n + col + lane];
                        }
                    }
                }

                if (col + width <= opt.n) {
                    result.copy_to(c + row * opt.n + col,
                                   sycl::ext::intel::esimd::element_aligned_tag{});
                } else {
                    for (int lane = 0; lane < static_cast<int>(opt.n - col); ++lane) {
                        c[row * opt.n + col + lane] = result[lane];
                    }
                }
            };

            return queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::nd_range<1>((groups + workgroup - 1) / workgroup * workgroup,
                                                       workgroup),
                                     kernel);
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
        validate(hc.data(), make_reference(ha, hb, opt.m, opt.n, opt.k), count_c);

        const double seconds = median(times);
        const double gflops = 2.0 * opt.m * opt.n * opt.k / seconds / 1e9;
        std::cout << std::fixed << std::setprecision(3)
                  << "ESIMD GEMM: " << opt.m << "x" << opt.k << " * "
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
