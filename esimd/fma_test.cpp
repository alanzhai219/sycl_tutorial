#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <iostream>

inline auto createExceptionHandler() {
  return [](sycl::exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << "sycl::exception: " << e0.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "std::exception: " << e.what() << std::endl;
      } catch (...) {
        std::cout << "generic exception\n";
      }
    }
  };
}

int main() {
  constexpr unsigned Size = 128;
  constexpr unsigned VL = 16; // fma just support 16 width
  int err_cnt = 0;

  try {
    sycl::queue q(sycl::gpu_selector_v, createExceptionHandler());
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<sycl::info::device::name>() << "\n";

    float *h_a = static_cast<float*>(malloc(Size * sizeof(float)));
    float *h_b = static_cast<float*>(malloc(Size * sizeof(float)));
    float *h_c = static_cast<float*>(malloc(Size * sizeof(float)));
    float *h_d = static_cast<float*>(malloc(Size * sizeof(float)));

    float *a = sycl::malloc_device<float>(Size, q); // USM memory for A
    float *b = sycl::malloc_device<float>(Size, q);
    float *c = sycl::malloc_device<float>(Size, q);
    float *d = sycl::malloc_device<float>(Size, q);

    for (unsigned i = 0; i < Size; i++) {
      h_a[i] = i;
      h_b[i] = 2.0f;
      h_c[i] = 0.3f;
      h_d[i] = 0.0f;
    }

    q.memcpy(a, h_a, Size * sizeof(float)).wait();
    q.memcpy(b, h_b, Size * sizeof(float)).wait();
    q.memcpy(c, h_c, Size * sizeof(float)).wait();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(Size / VL, [=](sycl::id<1> i) [[intel::sycl_explicit_simd]] {
        auto element_offset = i * VL;

        sycl::ext::intel::esimd::simd<float, VL> vec_a;
        vec_a.copy_from(a + element_offset); // Pointer arithmetic uses element offset

        sycl::ext::intel::esimd::simd<float, VL> vec_b;
        vec_b.copy_from(b + element_offset); // accessor API uses byte-offset

        sycl::ext::intel::esimd::simd<float, VL> vec_c;

        vec_c.copy_from(c + element_offset);
        sycl::ext::intel::esimd::simd<float, VL> vec_d = sycl::ext::intel::experimental::esimd::fma(vec_a, vec_b, vec_c);
        vec_d.copy_to(d + element_offset);
      });
    }).wait_and_throw();

    q.memcpy(h_d, d, Size * sizeof(float)).wait();

    for (unsigned i = 0; i < Size; ++i) {
      if (h_d[i] != (h_a[i] * h_b[i] + h_c[i])) {
        err_cnt++;
        std::cout << "failed at " << i << ": " << h_c[i] << " != " << h_a[i] << " + " << h_b[i] << std::endl;
      }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    sycl::free(d, q);
  } catch (sycl::exception &e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    return 1;
  }
  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}