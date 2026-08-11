#pragma once
#include "gpu_defs.h"
#include <assert.h>

// Convenience macro for checking CUDA/cuFFT/NCCL runtime API results; wrap it around any runtime
// API call. Aborts on error, and is always active -- it is not gated on release builds.
#define checkCuda(val)           __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )

template <typename T>
inline T __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
  if (code) {
    fprintf(stderr, "CUDA error: %s (code=%d)  \"%s\" at %s:%d \n", cudaGetErrorString(code), (unsigned int)code, func, file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
  return code;
}

template <>
inline cufftResult __checkCudaErrors__(cufftResult code, const char *func, const char *file, int line) 
{
  if (code) {
    fprintf(stderr, "CUDA error: (code=%d)  \"%s\" at %s:%d \n", (unsigned int)code, func, file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
  return code;
}

template <>
inline ncclResult_t __checkCudaErrors__(ncclResult_t code, const char *func, const char *file, int line) 
{
  if (code) {
    fprintf(stderr, "CUDA error: (code=%d)  \"%s\" at %s:%d \n", (unsigned int)code, func, file, line);
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
  return code;
}
 
inline cudaError_t print_cudims(dim3 dimGrid, dim3 dimBlock) {
  printf("dimBlock: %d %d %d dimGrid: %d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);
  return checkCuda(cudaGetLastError());
}
