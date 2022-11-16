extern "C" __global__ void add(
    const int*   x,
    const int*   y,
          int*   z,
          int  len
) {
    for (
        int i = blockDim.x * blockIdx.x + threadIdx.x;
            i < len;
            i += gridDim.x * blockDim.x
    ) {
        z[i] = x[i] + y[i];
    }
}

