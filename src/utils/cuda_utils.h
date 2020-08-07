
#ifndef _CUDA_UTILS
#define _CUDA_UTILS
#include <cuda_runtime.h>
#include <iostream>
inline void CudaCheck(cudaError_t error, const char *file_name, int line) {
  if(error != cudaSuccess)
    std::cout << "CUDA error " << cudaGetErrorString(error) << " at " << file_name << ":" << line;
}
#define CUDA_CHECK(error) CudaCheck((error), __FILE__, __LINE__)
#endif
