```bash
mkdir build
cd build
cmake ..
make
# icpx -fsycl -shared -fPIC wrapper.cpp vector_add.cpp -o libsycl_wrapper.so
# g++ main.cpp -o main -Llibsycl_wrapper.so
```
