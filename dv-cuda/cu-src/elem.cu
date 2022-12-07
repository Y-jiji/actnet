#ifndef __NAME__
#define __NAME__ add
#endif
#ifndef __TYPE__
#define __TYPE__ float
#endif
#ifndef __OP__
#define __OP__ +
#endif

extern "C"
__global__ void __NAME__(
    __TYPE__ * __restrict__ a,
    __TYPE__ * __restrict__ b,
    __TYPE__ * __restrict__ c,
    size_t bat_a,
    size_t bat_b
) {
    const size_t STEP = ( blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z );
    const size_t ID = ( (((( blockIdx.x * blockDim.y + blockIdx.y ) * blockDim.z + blockIdx.z) * gridDim.x + threadIdx.x) * gridDim.y + threadIdx.y) * gridDim.z + threadIdx.z );
    if (bat_a % bat_b == 0 && ID < bat_a) {
        size_t ISTEP = bat_b * ((bat_b < STEP) ? 1 : STEP/bat_b);
        size_t JSTEP = (bat_b < STEP) ? bat_b : STEP;
        size_t ISTART = ID / JSTEP;
        size_t JSTART = ID % JSTEP;
        size_t IEND = ISTART + ISTEP + (bat_a/ISTEP)*bat_b;
        size_t JEND = JSTART + JSTEP + bat_b/JSTEP;
        for (size_t i = ISTART; i != IEND; i += ISTEP) {
            for (size_t j = JSTART; j != JEND; j += JSTEP) {
                c[i + j] = a[i + j] __OP__ b[j];
            }
        }
    } else if (bat_b % bat_a == 0 && ID < bat_b) {
        size_t ISTEP = bat_a * ((bat_a < STEP) ? 1 : STEP/bat_a);
        size_t JSTEP = (bat_a < STEP) ? bat_a : STEP;
        size_t ISTART = ID / JSTEP;
        size_t JSTART = ID % JSTEP;
        size_t IEND = ISTART + ISTEP + (bat_b/ISTEP)*bat_a;
        size_t JEND = JSTART + JSTEP + bat_a/JSTEP;
        for (size_t i = 0; i != IEND; i += ISTEP) {
            for (size_t j = 0; j != JEND; j += JSTEP) {
                c[i + j] = a[j] __OP__ b[i + j];
            }
        }
    }
}

#undef __NAME__
#undef __TYPE__
#undef __OP__