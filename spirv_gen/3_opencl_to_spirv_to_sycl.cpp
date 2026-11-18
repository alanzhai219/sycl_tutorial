#include <ocloc_api.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

std::string kOpenclVectorAdd = R"CLC(
__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
)CLC";

std::vector<std::byte> compile_opencl_to_spirv(const std::string& source) {
    const char* args[] = {
        "ocloc",
        "-spv_only",
        "-file",
        "vector_add.cl"
    };

    const uint8_t* src_data[] = {reinterpret_cast<const uint8_t*>(source.c_str())};
    const uint64_t src_len[] = {static_cast<uint64_t>(source.size() + 1)};
    const char* src_name[] = {"vector_add.cl"};

    uint32_t num_outputs = 0;
    uint8_t** outputs = nullptr;
    uint64_t* output_sizes = nullptr;
    char** output_names = nullptr;

    int ret = oclocInvoke(
        static_cast<uint32_t>(sizeof(args) / sizeof(args[0])),
        args,
        1,
        src_data,
        src_len,
        src_name,
        0,
        nullptr,
        nullptr,
        nullptr,
        &num_outputs,
        &outputs,
        &output_sizes,
        &output_names);

    if (ret != 0) {
        throw std::runtime_error("oclocInvoke failed with code: " + std::to_string(ret));
    }

    std::vector<std::byte> spirv;
    for (uint32_t i = 0; i < num_outputs; ++i) {
        std::string name = output_names[i] ? output_names[i] : "";
        if (name.find(".spv") != std::string::npos) {
            auto* p = reinterpret_cast<std::byte*>(outputs[i]);
            spirv.assign(p, p + output_sizes[i]);
            break;
        }
    }

    oclocFreeOutput(&num_outputs, &outputs, &output_sizes, &output_names);

    if (spirv.empty()) {
        throw std::runtime_error("No SPIR-V output from ocloc");
    }

    return spirv;
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

int main() {
    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

        auto spirv = compile_opencl_to_spirv(kOpenclVectorAdd);
        std::cout << "Compiled OpenCL C to SPIR-V: " << spirv.size() << " bytes\n";

        auto src_bundle = syclex::create_kernel_bundle_from_source(
            q.get_context(), syclex::source_language::spirv, spirv);
        auto exec_bundle = syclex::build(src_bundle);
        auto kernel = exec_bundle.ext_oneapi_get_kernel("vector_add");

        const size_t n = 1024;
        std::vector<float> a(n), b(n), c(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(3 * i);
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

        std::cout << "E2E (OpenCL C -> SPIR-V -> SYCL) passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
