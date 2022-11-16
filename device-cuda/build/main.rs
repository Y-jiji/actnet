use std::error::Error;

mod bind_cuda;
mod bind_nvptx;

use bind_cuda::*;
// use bind_nvptx::*;

fn main() -> Result<(), Box<dyn Error>> {
    bind_cuda("cuda_runtime.h", "cuda-rt.rs")?;
    bind_cuda("cuda.h", "cuda-driver.rs")?;
    bind_cuda("cublas.h", "cublas.rs")?;
    Ok(())
}