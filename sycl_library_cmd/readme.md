```bash
clang++ -fsycl -fPIC -shared  vector_add.cpp -o libvec.so
g++ main.cpp -I/my/local/dpcpp/include -L/my/local/dpcpp/lib -lsycl -lvec -L.
```
