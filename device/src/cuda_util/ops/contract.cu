#define UL unsigned
#define TILE_SIZE 4
#define MIN(a, b) ((a)<(b)?(a):(b))

// don't support gridDim.z
extern "C" __global__ void
contract(
    // base pointers of input
    const float const* a, const float const* b, float *const c,
    // offset length of each index
    UL ia_len, UL ja_len, UL k_len,
    UL ib_len, UL jb_len
) {
    // ia = ia_bat*TILE_SIZE + ia_off
    // ja = ja_bat*TILE_SIZE + ja_off
    // analogously for ib, jb, k, elided
    __shared__
    float tile_a[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    __shared__
    float tile_b[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    __shared__
    float tile_c[TILE_SIZE][TILE_SIZE][TILE_SIZE][TILE_SIZE];

    UL shf = 0;
    UL msk = gridDim.z;
    msk = msk ? ((gridDim.z >> shf) >> 16) : (msk >> 16);
    shf += (msk != 0) << 4;
    msk = msk ? ((gridDim.z >> shf) >> 8) : (msk >> 8);
    shf += (msk != 0) << 3;
    msk = msk ? ((gridDim.z >> shf) >> 4) : (msk >> 4);
    shf += (msk != 0) << 2;
    msk = msk ? ((gridDim.z >> shf) >> 2) : (msk >> 2);
    shf += (msk != 0) << 1;
    shf >>= 1;
    msk = (UINT32_MAX) ^ (UINT32_MAX << shf);

    for (int ia_bat = blockIdx.x; ia_bat < ia_len/TILE_SIZE; ia_bat += gridDim.x) {
        for (int ib_bat = blockIdx.y; ib_bat < ib_len/TILE_SIZE; ib_bat += gridDim.y) {
            for (int ja_bat = (blockIdx.z & msk); ja_bat < ja_len/TILE_SIZE; ja_bat += (1 << shf)) {
                for (int jb_bat = (blockIdx.z >> shf); jb_bat < jb_len/TILE_SIZE; jb_bat += (gridDim.z >> shf)) {
                    // initialize the tile for temporary values in tensor c (copy back after computation)
                    for (int ia_off = threadIdx.x; ia_off < ia_len - ia_bat*TILE_SIZE; ia_off+=blockDim.x) {
                        for (int ja_off = threadIdx.y; ja_off < MIN(ja_len-ja_bat*TILE_SIZE, TILE_SIZE); ja_off+=blockDim.y) {
                            for (int ib_off = threadIdx.z; ib_off < MIN(ib_len-ib_bat*TILE_SIZE, TILE_SIZE); ib_off+=threadIdx.z) {
                                for (int jb_off = 0; jb_off < MIN(jb_len-jb_bat*TILE_SIZE, TILE_SIZE); ++jb_off) {
                                    tile_c[ia_off][ib_off][ja_off][jb_off] = 0;
                                }
                            }
                        }
                    }

                    // $litte_cache_miss_rate * k_len * TILE_SIZE^4$
                    for (int k_bat = 0; k_bat < k_len/TILE_SIZE; ++k_bat) {
                        // $big_cache_miss_rate * TILE_SIZE^3$
                        /* ------------ copy data into shared memory, for higher cache performance ------------ */
                        if (threadIdx.x & 1 == 0) {
                            const float const* a_bat = a + ((ia_bat*k_len+k_bat)*ja_len+ja_bat)*TILE_SIZE;
                            for (int ia_off = threadIdx.x >> 1; ia_off < MIN(ia_len-ia_bat*TILE_SIZE, TILE_SIZE); ia_off += ((blockDim.x+1)>>1)) {
                                for (int k_off = threadIdx.y; k_off < MIN(ja_len-ja_bat*TILE_SIZE, TILE_SIZE); k_off += blockDim.y) {
                                    for (int ja_off = threadIdx.z; ja_off < MIN(ja_len-ja_bat*TILE_SIZE, TILE_SIZE); ja_off += blockDim.z) {
                                        tile_a[ia_off][ja_off][k_off] = a_bat[(ia_off*k_len+k_off)*ja_len+ja_off];
                                    }
                                }
                            }
                        } else {
                            const float const* b_bat = b + ((ib_bat*k_len+k_bat)*jb_len+jb_bat)*TILE_SIZE;
                            for (int ib_off = threadIdx.x >> 1; ib_off < MIN(ib_len-ib_bat*TILE_SIZE, TILE_SIZE); ib_off += (blockDim.x>>1)) {
                                for (int k_off = threadIdx.y; k_off < MIN(jb_len-jb_bat*TILE_SIZE, TILE_SIZE); k_off += blockDim.y) {
                                    for (int jb_off = threadIdx.z; jb_off < MIN(jb_len-jb_bat*TILE_SIZE, TILE_SIZE); jb_off += blockDim.z) {
                                        tile_b[ib_off][jb_off][k_off] = b_bat[(ib_off*k_len+k_off)*jb_len+jb_off];
                                    }
                                }
                            }
                        }
                        __syncthreads(); // synchronize threads in this grid, i.e. wait all threads in this block execute to this point

                        // $small_cache_miss_time * TILE_SIZE^5$
                        /* --------------------- perform tensor contraction on each tile --------------------- */
                        for (int ia_off = threadIdx.x; ia_off < MIN(ia_len - ia_bat*TILE_SIZE, TILE_SIZE); ia_off+=blockDim.x) {
                            for (int ja_off = threadIdx.y; ja_off < MIN(ja_len-ja_bat*TILE_SIZE, TILE_SIZE); ja_off+=blockDim.y) {
                                for (int ib_off = threadIdx.z; ib_off < MIN(ib_len-ib_bat*TILE_SIZE, TILE_SIZE); ib_off+=threadIdx.z) {
                                    for (int jb_off = 0; jb_off < MIN(jb_len-jb_bat*TILE_SIZE, TILE_SIZE); ++jb_off) {
                                        for (int k_off = 0; k_off < MIN(k_len-k_bat*TILE_SIZE, TILE_SIZE); ++k_off) {
                                            tile_c[ia_off][ja_off][ib_off][jb_off] += 
                                                tile_a[ia_off][ja_off][k_off] * tile_b[ib_off][jb_off][k_off];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __syncthreads();

                    float* c_bat = c + (((ia_bat*ja_len+ja_bat)*ib_len+ib_bat)*jb_len+jb_bat)*TILE_SIZE;
                    // $big_cache_miss_rate * TILE_SIZE^4$
                    for (int ia_off = threadIdx.x; ia_off < MIN(ia_len - ia_bat*TILE_SIZE, TILE_SIZE); ia_off+=blockDim.x) {
                        for (int ja_off = threadIdx.y; ja_off < MIN(ja_len-ja_bat*TILE_SIZE, TILE_SIZE); ja_off+=blockDim.y) {
                            for (int ib_off = threadIdx.z; ib_off < MIN(ib_len-ib_bat*TILE_SIZE, TILE_SIZE); ib_off+=threadIdx.z) {
                                for (int jb_off = 0; jb_off < MIN(jb_len-jb_bat*TILE_SIZE, TILE_SIZE); ++jb_off) {
                                    c_bat[((ia_off*ja_len+ja_off)*ib_len+ib_off)*jb_len+jb_off] = tile_c[ia_off][ja_off][ib_off][jb_off];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#undef UL