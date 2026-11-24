#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    try {
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) throw std::runtime_error("No SYCL GPU found");
        if (argc == 1) {
            std::cout << "Usage: " << argv[0] << " DEVICE_INDEX [MiB_PER_ARRAY=256]\n";
            for (std::size_t i = 0; i < devices.size(); ++i) {
                std::cout << i << ": " << devices[i].get_info<sycl::info::device::name>()
                          << " | " << devices[i].get_platform().get_info<sycl::info::platform::name>() << '\n';
            }
            return 0;
        }
        if (argc > 3) throw std::invalid_argument("Too many arguments");
        auto integer = [](const std::string & text) {
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("Expected a non-negative integer");
            }
            return std::stoull(text);
        };
        const auto index = integer(argv[1]);
        const auto size_mib = argc == 3 ? integer(argv[2]) : 256;
        if (index >= devices.size() || size_mib == 0 || size_mib > 16384) {
            throw std::invalid_argument("Invalid device index or array size (1..16384 MiB)");
        }
        const auto device = devices[index];
        sycl::queue queue(device, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        const std::size_t bytes = static_cast<std::size_t>(size_mib) * 1024 * 1024;
        const std::size_t n = bytes / sizeof(float);
        if (bytes > device.get_info<sycl::info::device::max_mem_alloc_size>() ||
            bytes > device.get_info<sycl::info::device::global_mem_size>() / 4) {
            throw std::invalid_argument("Arrays too large for this device");
        }
        float * a = nullptr;
        float * b = nullptr;
        float * c = nullptr;
        try {
            a = sycl::malloc_device<float>(n, queue);
            b = sycl::malloc_device<float>(n, queue);
            c = sycl::malloc_device<float>(n, queue);
            if (!a || !b || !c) throw std::runtime_error("Device allocation failed");
            queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const std::size_t i = id[0];
                a[i] = 0.0f;
                b[i] = static_cast<float>(i % 1024);
                c[i] = 1.0f;
            }).wait_and_throw();
            std::cout << "Device: " << device.get_info<sycl::info::device::name>()
                      << "\nArrays: 3 x " << size_mib << " MiB; 5 warmups, 30 measured runs\n"
                      << "Effective GB/s, not measured DRAM traffic. GPU event timing only.\n";

            auto measure = [&](const char * name, int streams, float offset, auto submit) {
                for (int i = 0; i < 5; ++i) submit().wait_and_throw();
                std::vector<double> times;
                for (int i = 0; i < 30; ++i) {
                    sycl::event event = submit();
                    event.wait_and_throw();
                    const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    if (end <= start) throw std::runtime_error("Invalid GPU timestamps");
                    times.push_back((end - start) * 1e-9);
                }
                std::sort(times.begin(), times.end());
                const double seconds = (times[14] + times[15]) / 2;

                // Validate outside the timed interval, using a small host buffer.
                std::vector<float> host(std::min(n, std::size_t{262144}));
                for (std::size_t base = 0; base < n; base += host.size()) {
                    const std::size_t count = std::min(host.size(), n - base);
                    queue.memcpy(host.data(), a + base, count * sizeof(float)).wait_and_throw();
                    for (std::size_t j = 0; j < count; ++j) {
                        const float expected = static_cast<float>((base + j) % 1024) + offset;
                        if (!std::isfinite(host[j]) || host[j] != expected) {
                            throw std::runtime_error(std::string(name) + " validation failed");
                        }
                    }
                }
                std::cout << std::fixed << std::setprecision(3) << name
                          << ": " << seconds * 1000 << " ms, "
                          << static_cast<double>(bytes) * streams / seconds / 1e9 << " GB/s, PASS\n";
            };
            measure("Copy  1r+1w", 2, 0.0f, [&] {
                return queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = b[i]; });
            });
            measure("Add   2r+1w", 3, 1.0f, [&] {
                return queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = b[i] + c[i]; });
            });
            measure("Triad 1r+2w", 3, 2.0f, [&] {
                return queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = b[i] + 2.0f * c[i]; });
            });
        } catch (...) {
            if (a) sycl::free(a, queue);
            if (b) sycl::free(b, queue);
            if (c) sycl::free(c, queue);
            throw;
        }
        sycl::free(a, queue);
        sycl::free(b, queue);
        sycl::free(c, queue);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
