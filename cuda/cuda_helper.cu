#include <iostream>
#include <stdio.h>

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        auto error = cudaGetErrorString(result);
        std::cout << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        std::cout << error << '\n';
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}