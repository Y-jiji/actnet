use std::error::Error;

mod bind_cuda;
mod bind_nvptx;

use bind_cuda::*;
use bind_nvptx::*;

fn main() -> Result<(), Box<dyn Error>> {
    bind_cuda("cuda_runtime.h", "cuda-rt.rs", "cudart")?;
    bind_cuda("cuda.h", "cuda-driver.rs", "cuda")?;
    bind_cuda("cublas.h", "cublas.rs", "cublas")?;
    let mut children_proc = Vec::new();
    for cu in detect_cufile()? 
    { children_proc.push(nvcc_compile(cu.as_str())?); }
    for mut ch in children_proc
    { ch.wait()?; }
    println!("rerun-if-changed=build/main.rs");
    println!("rerun-if-changed=cu-src/*");
    // panic!("ok");
    Ok(())
}