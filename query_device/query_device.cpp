#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

struct devProp {
  // compute
  // slice
  size_t slc_count                  = -1;
  size_t xc_count_per_slc           = -1;
  // xc
  size_t xve_count_per_xc           = -1;
  // xve
  size_t xve_simd_width             = -1;
  size_t xve_count                  = -1;
  // memory
  // global
  size_t global_mem_size            = -1;
  size_t global_mem_cache_size      = -1;
  size_t global_mem_cache_line_size = -1;
  // local
  size_t local_mem_size             = -1;
  // work group
  size_t max_work_group_size        = -1;
  // sub group
  size_t max_sub_group_size         = -1;
  std::vector<size_t> sub_group_size_range = {};
};

void queryDevice(sycl::device d, devProp* p) {
  p->slc_count = d.get_info<sycl::ext::intel::info::device::gpu_slices>();
  p->xc_count_per_slc = d.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  p->xve_count_per_xc = d.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>();
  p->xve_count = d.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
  p->xve_simd_width = d.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
  p->global_mem_size = d.get_info<sycl::info::device::global_mem_size>();
  p->local_mem_size = d.get_info<sycl::info::device::local_mem_size>();
  p->sub_group_size_range = d.get_info<sycl::info::device::sub_group_sizes>();
  p->max_sub_group_size = d.get_info<sycl::info::device::max_num_sub_groups>();
}

int main() {
  for (auto& p : sycl::platform::get_platforms()) {
    std::cout << "SYCL Platform: "
              << p.get_info<sycl::info::platform::name>()
              << " is associated with SYCL Backend: "
              << p.get_backend() << std::endl;
  }

  auto plt = sycl::platform(sycl::gpu_selector_v);
  auto dev = plt.get_devices(sycl::info::device_type::gpu)[0];

  devProp p;
  queryDevice(dev, &p);

  std::cout << "# global memory size(GB)  : " << (p.global_mem_size >> 30) << "\n";
  std::cout << "# local memory size(KB)   : " << (p.local_mem_size >> 10) << "\n";
  std::cout << "# slice count             : " << p.slc_count << "\n";
  std::cout << "# xc count per slice      : " << p.xc_count_per_slc << "\n";
  std::cout << "# xve count per xc        : " << p.xve_count_per_xc << "\n";
  std::cout << "# xve count               : " << (p.xve_count) << "\n";
  std::cout << "# xve simd width(element) : " << (p.xve_simd_width) << "\n";

  return 0;
}
