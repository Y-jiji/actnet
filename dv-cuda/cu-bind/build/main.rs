use std::error::Error;

mod bind_cuda;

use bind_cuda::*;

fn main() -> Result<(), Box<dyn Error>> {
    bind_cuda("cuda_runtime.h", "rt.rs", "cudart")?;
    bind_cuda("cuda.h", "drv.rs", "cuda")?;
    bind_cuda("cublas.h", "blas.rs", "cublas")?;
    println!("rerun-if-env-changed=CUDA_INCLUDE_PATH");
    println!("rerun-if-env-changed=CUDA_LIBRARY_PATH");
    Ok(())
}