#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

std::vector<std::byte> read_spirv(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open SPIR-V file: " + path);
    }

    in.seekg(0, std::ios::end);
    const auto size = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<std::byte> data(size);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!in.good() && !in.eof()) {
        throw std::runtime_error("Failed to read SPIR-V file: " + path);
    }
    return data;
}

bool verify(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c) {
    constexpr float kEps = 1e-6f;
    for (size_t i = 0; i < a.size(); ++i) {
        float expected = a[i] + b[i];
        if (std::fabs(c[i] - expected) > kEps) {
            std::cerr << "Mismatch at i=" << i << ": got " << c[i] << ", expected " << expected << "\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string spv_path = (argc > 1) ? argv[1] : "vector_add.spv";

    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        const size_t n = 1024;
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(2 * i);
        }

        float* a_usm = sycl::malloc_shared<float>(n, q);
        float* b_usm = sycl::malloc_shared<float>(n, q);
        float* c_usm = sycl::malloc_shared<float>(n, q);
        if (!a_usm || !b_usm || !c_usm) {
            throw std::runtime_error("USM allocation failed");
        }
        for (size_t i = 0; i < n; ++i) {
            a_usm[i] = a[i];
            b_usm[i] = b[i];
            c_usm[i] = 0.0f;
        }

        auto spirv = read_spirv(spv_path);

        auto src_bundle = syclex::create_kernel_bundle_from_source(
            q.get_context(), syclex::source_language::spirv, spirv);
        auto exec_bundle = syclex::build(src_bundle);
        auto kernel = exec_bundle.ext_oneapi_get_kernel("vector_add");

        q.submit([&](sycl::handler& h) {
            h.set_arg(0, a_usm);
            h.set_arg(1, b_usm);
            h.set_arg(2, c_usm);
            h.set_arg(3, static_cast<int>(n));
            h.parallel_for(sycl::range<1>(n), kernel);
        });
        q.wait_and_throw();

        for (size_t i = 0; i < n; ++i) {
            c[i] = c_usm[i];
        }

        sycl::free(a_usm, q);
        sycl::free(b_usm, q);
        sycl::free(c_usm, q);

        if (!verify(a, b, c)) {
            return 2;
        }

        std::cout << "SYCL loaded SPIR-V successfully, vector add passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
