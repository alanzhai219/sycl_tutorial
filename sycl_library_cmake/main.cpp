#include <iostream>
#include "wrapper.hpp"

int main() {
    const int N = 16;
    // Allocate device memory for two vectors of size 16
    int* a = static_cast<int*>(malloc(sizeof(int) * N));
    int* b = static_cast<int*>(malloc(sizeof(int) * N));
    int* c = static_cast<int*>(malloc(sizeof(int) * N));

    // Initialize the vectors with random values
    for (int i=0; i<N; i++) {
        a[i]=rand()%100;
        b[i]=rand()%100;
        c[i]=rand()%100;
    }

    wrapper_vector_add(a,b,c,N);
    // Print the results using cudaGetLastError function
    std::cout<<"All done"<<std::endl;
    for (int i=0; i<N; i++) {
        printf("a[%d] + b[%d] = %d\n", a[i], b[i], c[i]);
    }

    //
    free(a);
    free(b);
    free(c);

    return EXIT_SUCCESS;
}
