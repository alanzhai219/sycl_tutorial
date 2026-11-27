#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct options {
    std::size_t m = 256;
    std::size_t n = 256;
    std::size_t k = 256;
    int runs = 30;
};

inline options parse_options(int argc, char **argv) {
    options opt;
    if (argc > 5) throw std::invalid_argument("usage: PROGRAM [M=256] [N=256] [K=256] [RUNS=30]");
    auto integer = [](const char *text) {
        return std::stoull(text);
    };
    if (argc >= 2) opt.m = integer(argv[1]);
    if (argc >= 3) opt.n = integer(argv[2]);
    if (argc >= 4) opt.k = integer(argv[3]);
    if (opt.m == 0 || opt.n == 0 || opt.k == 0) throw std::invalid_argument("matrix dimensions must be positive");
    if (argc == 5) opt.runs = std::stoi(argv[4]);
    if (opt.runs < 1 || opt.runs > 10000) throw std::invalid_argument("runs must be in 1..10000");
    return opt;
}

inline std::uint32_t hash32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline float random_float(std::size_t i, std::uint32_t seed) {
    const std::uint32_t x = hash32(static_cast<std::uint32_t>(i) + seed);
    return static_cast<float>((x & 0xffffU)) / 65535.0f - 0.5f;
}

inline sycl::queue make_queue() {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) throw std::runtime_error("no SYCL GPU found");

    for (const auto &device : devices) {
        if (device.get_platform().get_info<sycl::info::platform::name>().find("Level-Zero") != std::string::npos) {
            return sycl::queue(device, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
        }
    }
    return sycl::queue(devices.front(), {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
}

inline void validate(const float *host_c, const std::vector<float> &reference, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(host_c[i]) || std::fabs(host_c[i] - reference[i]) > 1e-4f * std::max(1.0f, std::fabs(reference[i]))) {
            throw std::runtime_error("GEMM validation failed at element " + std::to_string(i));
        }
    }
}

inline std::vector<float> make_reference(const std::vector<float> &a, const std::vector<float> &b,
                                         std::size_t m, std::size_t n, std::size_t k) {
    std::vector<float> c(m * n, 0.0f);
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += static_cast<double>(a[row * k + inner]) * static_cast<double>(b[inner * n + col]);
            }
            c[row * n + col] = static_cast<float>(sum);
        }
    }
    return c;
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
