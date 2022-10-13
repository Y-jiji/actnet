#define BLOCK_ID (\
    blockIdx.x * gridDim.y * gridDim.z + \
    blockIdx.y * gridDim.z + \
    blockIdx.z\
)

#define THEAD_ID (\
    blockDim.x * blockDim.y * blockDim.z * BLOCK_ID + \
    threadIdx.x * blockDim.y * blockDim.z + \
    threadIdx.y * blockDim.z + \
    threadIdx.z\
)

#define STEP (\
    gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z)


#define ULL unsigned long long
extern "C" __global__ void
add_f32 (
    float* x, float* y, float* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] + y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
sub_f32 (
    float* x, float* y, float* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] - y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
mul_f32 (
    float* x, float* y, float* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] * y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
div_f32 (
    float* x, float* y, float* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] / y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
add_f64 (
    double* x, double* y, double* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] + y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
sub_f64 (
    double* x, double* y, double* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] - y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
mul_f64 (
    double* x, double* y, double* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] * y[i];
    }
}
#undef ULL


#define ULL unsigned long long
extern "C" __global__ void
div_f64 (
    double* x, double* y, double* z,
    ULL len
) {
    for (
        ULL i = THEAD_ID;
        i < len && i < i + STEP; 
        i += STEP
    ) {
        z[i] = x[i] / y[i];
    }
}
#undef ULL


#undef STEP
#undef THEAD_ID
#undef BLOCK_ID