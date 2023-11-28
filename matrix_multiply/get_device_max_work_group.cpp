#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
	
	auto platforms = platform::get_platforms();
	auto q = queue{ platforms[3].get_devices()[0] };  
	std::cout << "My Device: " << q.get_device().get_info<info::device::name>() << "\n";


	std::cout << "Max Compute Units: " << q.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
	std::cout << "Max Work Group Size: " << q.get_device().get_info<sycl::info::device::max_work_group_size>() << std::endl;
    return 0;

}
