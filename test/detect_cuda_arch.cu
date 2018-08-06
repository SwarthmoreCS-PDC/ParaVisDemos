#include <cuda_runtime.h>
#include <cstdio>

/*
Adapted from
https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
*/
int main() {
  cudaDeviceProp dP;
  int err = cudaGetDeviceProperties(&dP, 0);
  if(err != cudaSuccess) {
      cudaError_t error = cudaGetLastError();
      printf("CUDA error: %s", cudaGetErrorString(error));
      return err; /* Failure */
  }
  printf("-arch=sm_%d%d", dP.major, dP.minor);
  return 0; /* Success */
}
