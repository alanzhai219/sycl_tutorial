#include <sycl/sycl.hpp>
#include <iostream>

auto exception_handler = [] (sycl::exception_list exceptions) {
	for (const std::exception_ptr &e : exceptions) {
		try {
			std::rethrow_exception(e);
		}
		catch(const sycl::exception &e) {
			std::cout << "Caught SYCL exception: " << e.what() << "\n";
		}
		catch(const std::exception &e) {
			std::cout << "Caught std exception: " << e.what() << "\n";
		}
		catch(...) {
			std::cout << "Caught unknown exception\n";
		}
	}
};

void mykernel(sycl::id<1> idx, int* d_ptr){
    d_ptr[idx] = 1;
};


int main() {
    const size_t N = 100;
    // 1. create a queue
    sycl::queue q(sycl::gpu_selector_v, exception_handler);

    // 2. malloc memory on device
    int* ptr = static_cast<int*>(sycl::malloc_device(N, q));
    if (ptr == nullptr) {
      std::cout << "Allocation failed!" << std::endl;
    }

    // 3. launch kernel
    q.parallel_for(sycl::range<1>(N), [ptr](sycl::id<1> idx) { mykernel(idx, ptr); });

    // 4. free
    sycl::free(ptr, q);
    return 0;
}
