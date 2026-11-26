#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static float input_value(std::size_t i, std::uint32_t seed) {
    std::uint32_t x = static_cast<std::uint32_t>(i) + seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return sycl::bit_cast<float>(0x3f800000u | (x & 0x007fffffu));
}

int main(int argc, char ** argv) {
    try {
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) throw std::runtime_error("No SYCL GPU found");
        if (argc == 1) {
            std::cout << "Usage: " << argv[0]
                      << " DEVICE_INDEX [MiB=512] [RUNS=30]\n";
            for (std::size_t i = 0; i < devices.size(); ++i) {
                std::cout << i << ": " << devices[i].get_info<sycl::info::device::name>()
                          << " | " << devices[i].get_platform().get_info<sycl::info::platform::name>() << '\n';
            }
            return 0;
        }
        if (argc > 4) throw std::invalid_argument("Too many arguments");
        auto integer = [](const std::string & text) {
            if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
                throw std::invalid_argument("Expected a non-negative integer");
            }
            return std::stoull(text);
        };
        const auto index = integer(argv[1]);
        const auto size_mib = argc >= 3 ? integer(argv[2]) : 512;
        const auto runs = argc >= 4 ? integer(argv[3]) : 30;
        if (runs == 0 || runs > 10000) {
            throw std::invalid_argument("Invalid run count (1..10000)");
        }
        if (index >= devices.size() || size_mib == 0 || size_mib > 16384) {
            throw std::invalid_argument("Invalid device index or array size (1..16384 MiB)");
        }
        const auto device = devices[index];
        if (!device.has(sycl::aspect::usm_device_allocations) ||
            device.get_info<sycl::info::device::max_work_group_size>() < 256) {
            throw std::runtime_error("Device USM and work-group size 256 are required");
        }
        sycl::queue queue(device, [](sycl::exception_list errors) {
            for (const auto & error : errors) std::rethrow_exception(error);
        }, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        const std::size_t bytes = static_cast<std::size_t>(size_mib) * 1024 * 1024;
        const std::size_t n = bytes / sizeof(float);
        constexpr std::size_t local_size = 256;
        constexpr std::size_t elements_per_group = local_size * 4;
        const std::size_t groups = n / elements_per_group;
        const auto launch = sycl::nd_range<1>(sycl::range<1>(n / 4), sycl::range<1>(local_size));
        if (bytes > device.get_info<sycl::info::device::max_mem_alloc_size>() ||
            bytes > device.get_info<sycl::info::device::global_mem_size>() / 5) {
            throw std::invalid_argument("Arrays too large for this device");
        }
        float * data = nullptr;
        float * sums = nullptr;
        try {
            data = sycl::malloc_device<float>(n, queue);
            sums = sycl::malloc_device<float>(groups, queue);
            if (!data || !sums) throw std::runtime_error("Device allocation failed");
            queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                const std::size_t i = id[0];
                data[i] = input_value(i, 17);
            }).wait_and_throw();
            std::cout << "Device: " << device.get_info<sycl::info::device::name>()
                      << "\nPlatform: " << device.get_platform().get_info<sycl::info::platform::name>()
                      << "\nArray: " << size_mib << " MiB plus read checksums; 5 warmups, " << runs << " measured runs\n"
                      << "Effective decimal GB/s, not hardware-counted DRAM traffic. GPU event timing only.\n"
                      << "Read includes one FP32 checksum store per 1024 input elements.\n";
            const auto cache_bytes = device.get_info<sycl::info::device::global_mem_cache_size>();
            std::cout << "Reported global cache: " << cache_bytes / (1024.0 * 1024.0) << " MiB\n";
            if (cache_bytes == 0 || bytes / 8 <= cache_bytes) {
                std::cout << "WARNING: cache unknown or array <= 8x cache; increase size and check the bandwidth plateau.\n";
            }

            auto validate = [&](const char * name, const float * output, std::size_t count, auto expected, float tolerance = 0.0f) {
                std::vector<float> host(std::min(count, std::size_t{262144}));
                for (std::size_t base = 0; base < count; base += host.size()) {
                    const std::size_t chunk = std::min(host.size(), count - base);
                    queue.memcpy(host.data(), output + base, chunk * sizeof(float)).wait_and_throw();
                    for (std::size_t j = 0; j < chunk; ++j) {
                        const float reference = expected(base + j);
                        if (!std::isfinite(host[j]) || std::fabs(host[j] - reference) > tolerance * std::max(1.0f, std::fabs(reference))) {
                            throw std::runtime_error(std::string(name) + " validation failed at " + std::to_string(base + j));
                        }
                    }
                }
            };
            auto measure = [&](const char * name, auto submit, auto check) {
                for (int i = 0; i < 5; ++i) submit().wait_and_throw();
                std::vector<double> times;
                for (unsigned long long i = 0; i < runs; ++i) {
                    sycl::event event = submit();
                    event.wait_and_throw();
                    const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    if (end <= start) throw std::runtime_error("Invalid GPU timestamps");
                    times.push_back((end - start) * 1e-9);
                }
                std::sort(times.begin(), times.end());
                const double seconds = (times[times.size() / 2] + times[(times.size() - 1) / 2]) / 2;
                check();
                std::cout << std::fixed << std::setprecision(3) << name
                          << ": median=" << seconds * 1000 << " ms"
                          << ", bandwidth=" << bytes / seconds / 1e9 << " GB/s, PASS\n";
            };
            measure("Read", [&] {
                return queue.parallel_for(launch, [=](sycl::nd_item<1> it) {
                    sycl::float4 value;
                    value.load(it.get_global_linear_id(), data);
                    const float sum = value[0] + value[1] + value[2] + value[3];
                    const float group_sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                    if (it.get_local_linear_id() == 0) sums[it.get_group_linear_id()] = group_sum;
                });
            }, [&] {
                validate("Read checksum", sums, groups, [](std::size_t group) {
                    double sum = 0;
                    for (std::size_t j = 0; j < elements_per_group; ++j) {
                        sum += input_value(group * elements_per_group + j, 17);
                    }
                    return static_cast<float>(sum);
                }, 1e-5f);
            });
            measure("Write", [&] {
                return queue.parallel_for(launch, [=](sycl::nd_item<1> it) {
                    const std::size_t i = it.get_global_linear_id();
                    const std::size_t base = i * 4;
                    sycl::float4(input_value(base, 101), input_value(base + 1, 101),
                                 input_value(base + 2, 101), input_value(base + 3, 101)).store(i, data);
                });
            }, [&] {
                validate("Write output", data, n, [](std::size_t i) { return input_value(i, 101); });
            });
        } catch (...) {
            // Finish pending commands before releasing their allocations.
            try { queue.wait_and_throw(); } catch (...) {}
            if (data) sycl::free(data, queue);
            if (sums) sycl::free(sums, queue);
            throw;
        }
        sycl::free(data, queue);
        sycl::free(sums, queue);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
