use std::error::Error;

mod bind_nvptx;

use bind_nvptx::*;

fn main() -> Result<(), Box<dyn Error>> {
    // detect and compile .cu files to .ptx
    let mut children_proc = Vec::new();
    for cu in detect_cufile()? 
    { children_proc.push(nvcc_compile(cu.as_str())?); }
    for mut ch in children_proc
    { ch.wait()?; }
    println!("rerun-if-changed=cu-src/*");
    Ok(())
}