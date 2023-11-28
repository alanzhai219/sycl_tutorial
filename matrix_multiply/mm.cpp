#include <sycl/sycl.hpp>
// #include <io.h>
#include <ctime>
#include <chrono>

using namespace sycl;


void mm_kernel(queue& q, std::vector<float>& matrix_a, std::vector<float>& matrix_b, std::vector<float>& matrix_c, size_t N, size_t M) {
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << " | WORK_GROUP_SIZE= " << M << "x" << M << "\n";

    //# Create buffers for matrices
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    //# Submit command groups to execute on device
    auto e = q.submit([&](handler& h) {
        //# Create accessors to copy buffers to the device
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

        //# Define size for ND-Range and work-group size
        range<2> global_size(N, N);
        range<2> work_group_size(M, M);

        //# Parallel Compute Matrix Multiplication
        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item) {
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
            });
        });
    host_accessor hc(c, read_only);

    //# print kernel compute duration from event profiling
    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
}
//# floating point error verification function
bool almost_equal(float a, float b) {
    float tolerance = 1e-6;
    float diff = fabs(a - b);
    a = fabs(a);
    b = fabs(b);
    float bigger = (b > a) ? b : a;
    if (diff <= bigger * tolerance) return true;
    return false;
}

int main(int argc, char* argv[]) {

    // the full ND-Range is NxN
    size_t N = 1024;
    
    // work-group is M
    size_t M = 16;  
    // why M = 32, will report error?
    
    //# Define queue with default device for offloading computation
    auto platforms = platform::get_platforms();
    for (auto& p : platforms) {
      std::cout << "SYCL Platform: "
                << p.get_info<info::platform::name>()
                << " is associated with SYCL Backend: "
                << p.get_backend() << std::endl;
    }

    auto q = queue{ platforms[3].get_devices()[0], property::queue::enable_profiling {} };
	std::cout << "My Device: " << q.get_device().get_info<info::device::name>() << "\n";
	std::cout << "Max Compute Units: " << q.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
	std::cout << "Max Work Group Size: " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;


    int VERIFY = 0;
    int PRINT_OUTPUT_MATRIX = 0;

 
    //# Define vectors for matrices
    std::vector<float> matrix_a(N * N);
    std::vector<float> matrix_b(N * N);
    std::vector<float> matrix_c(N * N);
    std::vector<float> matrix_d(N * N);

    //# Initialize matrices with values
    float v1 = 2.f;
    float v2 = 3.f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            matrix_a[i * N + j] = v1++;
            matrix_b[i * N + j] = v2++;
            matrix_c[i * N + j] = 0.f;
            matrix_d[i * N + j] = 0.f;
        }

   


    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size   : " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";

    //# get start time
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    //# Call matrix multiplication kernel implementation
    mm_kernel(q, matrix_a, matrix_b, matrix_c, N, M);

    //# print kernel compute duration from host
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    std::cout << "Compute Duration      : " << duration / 1e+9 << " seconds\n";

    //# Print Output if -p in cmd-line
    if (PRINT_OUTPUT_MATRIX) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << matrix_c[i * N + j] << " ";
            }
            std::cout << "\n";
        }
    }
    else {
        std::cout << " [0][0] = " << matrix_c[0] << "\n";
    }

    //# Compute local and compare with offload computation if -v in cmd-line
    if (VERIFY) {
        int fail = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    matrix_d[i * N + j] += matrix_a[i * N + k] * matrix_b[k * N + j];
                }
                if (!almost_equal(matrix_c[i * N + j], matrix_d[i * N + j])) fail = 1;
            }
        }
        if (fail == 1) {
            std::cout << "FAIL\n";
        }
        else {
            std::cout << "PASS\n";
        }
    }
    return 0;
}
