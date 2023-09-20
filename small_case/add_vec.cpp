#include <iostream>
#include <sycl/sycl.hpp>
 
void add_vectors(sycl::queue& queue, sycl::buffer<float>& a, sycl::buffer<float>& b, sycl::buffer<float>& c) {
   sycl::range n(a.size());
 
   queue.submit([&](sycl::handler& cgh) {
      auto in_a = a.get_access<sycl::access::mode::read>(cgh);
      auto in_b = b.get_access<sycl::access::mode::read>(cgh);
      auto out_c = c.get_access<sycl::access::mode::write>(cgh);
 
      cgh.parallel_for<class add_vectors>(n, [=](sycl::id<1> i) {
               out_c[i] = in_a[i] + in_b[i];
      });
   });
}
 
int main(int, char**) {
   const size_t n = 100;
 
   std::vector<float> a(n, 1.0f);
   std::vector<float> b(n, 2.0f);
   std::vector<float> c(n, 0.0f);
 
   sycl::buffer<float> a_buf{a};
   sycl::buffer<float> b_buf{b};
   sycl::buffer<float> c_buf{c};
 
   sycl::queue q;
 
   add_vectors(q, a_buf, b_buf, c_buf);
 
   auto result = c_buf.get_access<sycl::access::mode::read>();
   for (size_t i = 0; i < n; ++i) {
      std::cout << result[i] << " ";
   }
 
   return 0;
}
