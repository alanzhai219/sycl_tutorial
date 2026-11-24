#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t mib = 1024 * 1024;

struct options {
    std::size_t device = 0;
    std::size_t array_mib = 256;
    std::size_t iterations = 50;
    std::size_t warmup = 5;
    std::size_t local = 256;
    bool list = false;
    bool help = false;
};

std::size_t number(const std::string & text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("Expected a non-negative integer: " + text);
    }
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::out_of_range("Integer exceeds size_t range");
    }
    return static_cast<std::size_t>(value);
}

options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            result.help = true;
        } else if (arg == "--list") {
            result.list = true;
        } else {
            std::size_t * value = nullptr;
            if (arg == "--device") value = &result.device;
            else if (arg == "--mib") value = &result.array_mib;
            else if (arg == "--iterations") value = &result.iterations;
            else if (arg == "--warmup") value = &result.warmup;
            else if (arg == "--local") value = &result.local;
            else throw std::invalid_argument("Unknown option: " + arg);
            if (++i == argc) throw std::invalid_argument("Missing value for " + arg);
            *value = number(argv[i]);
        }
    }
    if (result.array_mib == 0 || result.array_mib > std::numeric_limits<std::size_t>::max() / mib ||
        result.iterations < 3 || result.iterations > 100000 || result.warmup > 100000 || result.local == 0) {
        throw std::invalid_argument("Require MiB > 0, iterations in [3,100000], warmup <= 100000, local > 0");
    }
    return result;
}

void usage() {
    std::cout << "SYCL GPU effective memory bandwidth (FP32)\n"
              << "  --list                List GPU devices without allocating memory\n"
              << "  --device INDEX        GPU index from --list (default 0)\n"
              << "  --mib SIZE            MiB per array; three arrays (default 256)\n"
              << "  --iterations COUNT    Timed runs per operation (default 50, minimum 3)\n"
              << "  --warmup COUNT        Untimed runs per operation (default 5)\n"
              << "  --local SIZE          Work-items per work-group (default 256)\n";
}

float input_b(std::size_t i) { return 2.0f + static_cast<float>(i % 17) * 0.0625f; }
float input_c(std::size_t i) { return 3.0f + static_cast<float>(i % 13) * 0.0625f; }

enum class operation { copy, add, triad };

template<operation Op>
sycl::event launch(sycl::queue & queue, float * a, const float * b, const float * c,
                   std::size_t count, std::size_t global, std::size_t local) {
    return queue.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
        const std::size_t i = item.get_global_linear_id();
        if (i < count) {
            if constexpr (Op == operation::copy) a[i] = b[i];
            else if constexpr (Op == operation::add) a[i] = b[i] + c[i];
            else a[i] = b[i] + 2.0f * c[i];
        }
    });
}

template<operation Op>
void benchmark(const char * name, sycl::queue & queue, float * a, const float * b, const float * c,
               std::size_t count, std::size_t global, const options & opt) {
    for (std::size_t i = 0; i < opt.warmup; ++i) {
        launch<Op>(queue, a, b, c, count, global, opt.local).wait_and_throw();
    }

    std::vector<double> seconds;
    seconds.reserve(opt.iterations);
    for (std::size_t i = 0; i < opt.iterations; ++i) {
        sycl::event event = launch<Op>(queue, a, b, c, count, global, opt.local);
        event.wait_and_throw();
        const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
        if (end <= start) throw std::runtime_error("Invalid GPU profiling timestamps");
        seconds.push_back(static_cast<double>(end - start) * 1e-9);
    }

    // Validate every output element in bounded host-side chunks, outside the timed interval.
    std::vector<float> host(std::min(count, mib / sizeof(float)));
    for (std::size_t offset = 0; offset < count; offset += host.size()) {
        const std::size_t size = std::min(host.size(), count - offset);
        queue.memcpy(host.data(), a + offset, size * sizeof(float)).wait_and_throw();
        for (std::size_t j = 0; j < size; ++j) {
            const std::size_t i = offset + j;
            float expected = input_b(i);
            if constexpr (Op == operation::add) expected += input_c(i);
            if constexpr (Op == operation::triad) expected += 2.0f * input_c(i);
            if (!std::isfinite(host[j]) || std::abs(host[j] - expected) > 1e-6f) {
                throw std::runtime_error(std::string(name) + " validation failed at element " + std::to_string(i));
            }
        }
    }

    std::sort(seconds.begin(), seconds.end());
    const std::size_t mid = seconds.size() / 2;
    const double median = seconds.size() % 2 ? seconds[mid] : (seconds[mid - 1] + seconds[mid]) / 2;
    const double bytes = static_cast<double>(count) * sizeof(float) * (Op == operation::copy ? 2 : 3);
    std::cout << std::left << std::setw(10) << name << std::right << std::fixed << std::setprecision(3)
              << std::setw(14) << median * 1e3
              << std::setw(14) << bytes / median / 1e9
              << std::setw(14) << bytes / seconds.front() / 1e9
              << std::setw(14) << bytes / seconds.back() / 1e9
              << "    PASS\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse(argc, argv);
        if (opt.help) {
            usage();
            return 0;
        }
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) throw std::runtime_error("No SYCL GPU device found");
        if (opt.list) {
            for (std::size_t i = 0; i < devices.size(); ++i) {
                std::cout << '[' << i << "] " << devices[i].get_info<sycl::info::device::name>()
                          << " | " << devices[i].get_platform().get_info<sycl::info::platform::name>()
                          << " | device USM: " << devices[i].has(sycl::aspect::usm_device_allocations)
                          << " | shared USM: " << devices[i].has(sycl::aspect::usm_shared_allocations) << '\n';
            }
            return 0;
        }
        if (opt.device >= devices.size()) throw std::invalid_argument("Device index out of range; use --list");
        const auto device = devices[opt.device];
        if (!device.has(sycl::aspect::usm_device_allocations) || !device.has(sycl::aspect::queue_profiling)) {
            throw std::runtime_error("Device USM and queue profiling are required");
        }
        if (opt.local > device.get_info<sycl::info::device::max_work_group_size>() ||
            opt.local > device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]) {
            throw std::invalid_argument("Local size exceeds device limit");
        }
        const std::size_t bytes = opt.array_mib * mib;
        if (bytes > device.get_info<sycl::info::device::max_mem_alloc_size>() ||
            bytes > device.get_info<sycl::info::device::global_mem_size>() / 4) {
            throw std::invalid_argument("Array too large: respect max allocation and keep three arrays <= 75% of device memory");
        }
        const std::size_t count = bytes / sizeof(float);
        if (opt.local - 1 > std::numeric_limits<std::size_t>::max() - count) {
            throw std::invalid_argument("Rounded global size overflows size_t");
        }
        const std::size_t global = ((count + opt.local - 1) / opt.local) * opt.local;
        sycl::queue queue(device, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        auto release = [&queue](float * ptr) { if (ptr) sycl::free(ptr, queue); };
        using allocation = std::unique_ptr<float, decltype(release)>;
        allocation a(sycl::malloc_device<float>(count, queue), release);
        allocation b(sycl::malloc_device<float>(count, queue), release);
        allocation c(sycl::malloc_device<float>(count, queue), release);
        if (!a || !b || !c) throw std::runtime_error("Device allocation failed; reduce --mib");
        float * ap = a.get();
        float * bp = b.get();
        float * cp = c.get();
        queue.parallel_for(sycl::nd_range<1>{global, opt.local}, [=](sycl::nd_item<1> item) {
            const std::size_t i = item.get_global_linear_id();
            if (i < count) {
                ap[i] = 0.0f;
                bp[i] = 2.0f + static_cast<float>(i % 17) * 0.0625f;
                cp[i] = 3.0f + static_cast<float>(i % 13) * 0.0625f;
            }
        }).wait_and_throw();

        std::cout << "Device: " << device.get_info<sycl::info::device::name>()
                  << "\nPlatform: " << device.get_platform().get_info<sycl::info::platform::name>()
                  << "\nDriver: " << device.get_info<sycl::info::device::driver_version>()
                  << "\nArrays: 3 x " << opt.array_mib << " MiB, elements/array: " << count
                  << "\nIterations: " << opt.iterations << ", warmup: " << opt.warmup << ", local: " << opt.local
                  << "\nGPU event timing; allocation, initialization and validation excluded."
                  << "\nEffective GB/s (1 GB = 1e9 bytes), NOT measured DRAM traffic.\n\n"
                  << std::left << std::setw(10) << "Operation" << std::right
                  << std::setw(14) << "Median ms" << std::setw(14) << "Median GB/s"
                  << std::setw(14) << "Best GB/s" << std::setw(14) << "Worst GB/s" << "    Validation\n";
        benchmark<operation::copy>("Copy", queue, ap, bp, cp, count, global, opt);
        benchmark<operation::add>("Add", queue, ap, bp, cp, count, global, opt);
        benchmark<operation::triad>("Triad", queue, ap, bp, cp, count, global, opt);
        queue.wait_and_throw();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}