#include <stdio.h>

void printDevProp(cudaDeviceProp devProp) {
  printf("%s\n", devProp.name);
  printf("Major revision number:         %d\n", devProp.major);
  printf("Minor revision number:         %d\n", devProp.minor);
  printf("Total global memory:           %zu", devProp.totalGlobalMem);
  printf(" bytes\n");
  printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
  printf("Total amount of shared memory per block: %zu\n",
         devProp.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", devProp.regsPerBlock);
  printf("Warp size:                     %d\n", devProp.warpSize);
  printf("Maximum memory pitch:          %zu\n", devProp.memPitch);
  printf("Total amount of constant memory:         %zu\n",
         devProp.totalConstMem);
  return;
}

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("-----------------------------------------------\n");
    printDevProp(prop);
  }
  printf("-----------------------------------------------\n");
}