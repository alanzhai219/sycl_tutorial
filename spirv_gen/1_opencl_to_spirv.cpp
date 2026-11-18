#include <ocloc_api.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

bool write_binary_file(const std::string& path, const uint8_t* data, uint64_t size) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    return out.good();
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_spv = (argc > 1) ? argv[1] : "vector_add.spv";

    const char* args[] = {
        "ocloc",
        "-spv_only",
        "-file",
        "vector_add.cl"
    };

    const uint8_t* src_data[] = {
        reinterpret_cast<const uint8_t*>(kOpenclVectorAdd.c_str())
    };
    const uint64_t src_len[] = {
        static_cast<uint64_t>(kOpenclVectorAdd.size() + 1)
    };
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
        std::cerr << "oclocInvoke failed, ret=" << ret << "\n";
        return 1;
    }

    const uint8_t* spv_data = nullptr;
    uint64_t spv_size = 0;

    for (uint32_t i = 0; i < num_outputs; ++i) {
        std::string name = output_names[i] ? output_names[i] : "";
        if (name.find(".spv") != std::string::npos) {
            spv_data = outputs[i];
            spv_size = output_sizes[i];
            break;
        }
    }

    if (!spv_data || spv_size == 0) {
        std::cerr << "No SPIR-V output found from ocloc\n";
        oclocFreeOutput(&num_outputs, &outputs, &output_sizes, &output_names);
        return 1;
    }

    if (!write_binary_file(out_spv, spv_data, spv_size)) {
        std::cerr << "Failed to write " << out_spv << "\n";
        oclocFreeOutput(&num_outputs, &outputs, &output_sizes, &output_names);
        return 1;
    }

    std::cout << "Wrote SPIR-V: " << out_spv << " (" << spv_size << " bytes)\n";

    oclocFreeOutput(&num_outputs, &outputs, &output_sizes, &output_names);
    return 0;
}
