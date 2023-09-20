#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  for (auto& p : platform::get_platforms()) {
    std::cout << "SYCL Platform: "
              << p.get_info<info::platform::name>()
              << " is associated with SYCL Backend: "
              << p.get_backend() << std::endl;
  }
  return 0;
}
