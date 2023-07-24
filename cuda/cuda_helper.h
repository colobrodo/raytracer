#pragma once

// from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);