#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <iostream>
#include <string>

namespace matrix = sycl::ext::oneapi::experimental::matrix;

const char *dtype_name(matrix::matrix_type type) {
    switch (type) {
        case matrix::matrix_type::bf16:   return "BF16";
        case matrix::matrix_type::fp16:   return "FP16";
        case matrix::matrix_type::tf32:   return "TF32";
        case matrix::matrix_type::fp32:   return "FP32";
        case matrix::matrix_type::fp64:   return "FP64";
        case matrix::matrix_type::sint8:  return "INT8";
        case matrix::matrix_type::sint16: return "INT16";
        case matrix::matrix_type::sint32: return "INT32";
        case matrix::matrix_type::sint64: return "INT64";
        case matrix::matrix_type::uint8:  return "UINT8";
        case matrix::matrix_type::uint16: return "UINT16";
        case matrix::matrix_type::uint32: return "UINT32";
        case matrix::matrix_type::uint64: return "UINT64";
    }
    return "UNKNOWN";
}

std::string dimension(std::size_t fixed, std::size_t maximum) {
    if (fixed != 0) return std::to_string(fixed);
    if (maximum != 0) return "[1.." + std::to_string(maximum) + "]";
    return "unspecified";
}

int main() {
    try {
        const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (devices.empty()) {
            std::cerr << "No visible SYCL GPU devices.\n";
            return 1;
        }

        bool failed = false;
        for (const auto &dev : devices) {
            std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << '\n'
                      << "Platform: " << dev.get_platform().get_info<sycl::info::platform::name>() << '\n'
                      << "Driver: " << dev.get_info<sycl::info::device::driver_version>() << '\n';
            try {
                const auto combinations = dev.get_info<
                    sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
                std::cout << "Supported combinations: " << combinations.size() << '\n';
                if (combinations.empty()) {
                    std::cout << "No joint_matrix combinations reported by this runtime.\n";
                }
                std::size_t index = 0;
                for (const auto &c : combinations) {
                    std::cout << "\nCombination " << ++index << '\n'
                              << "  input dtype: A=" << dtype_name(c.atype)
                              << ", B=" << dtype_name(c.btype) << '\n'
                              << "  accumulator dtype: C=" << dtype_name(c.ctype) << '\n'
                              << "  output dtype: D=" << dtype_name(c.dtype) << '\n'
                              << "  shape: M=" << dimension(c.msize, c.max_msize)
                              << ", N=" << dimension(c.nsize, c.max_nsize)
                              << ", K=" << dimension(c.ksize, c.max_ksize) << '\n';
                }
            } catch (const sycl::exception &error) {
                std::cerr << "Query failed: " << error.what() << '\n';
                failed = true;
            }
            std::cout << '\n';
        }
        std::cout << "Operation: D = A * B + C; A[M,K], B[K,N], C/D[M,N].\n"
                  << "Shapes describe tiles, not the full GEMM dimensions.\n"
                  << "Layout and subgroup requirements must also be satisfied.\n"
                  << "This queries runtime capabilities; it does not execute or benchmark XMX.\n";
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}