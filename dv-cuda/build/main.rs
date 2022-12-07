use std::error::Error;

mod bind_nvptx;

use bind_nvptx::*;

fn main() -> Result<(), Box<dyn Error>> {
    // detect and compile .cu files to .ptx
    let mut children_proc = Vec::new();
    for (file, suffix, flag) in detect_cufile()? {
        let flag = flag.split('|').map(|s| s.trim()).filter(|x| *x!="").collect();
        children_proc.push((file.clone()+&suffix, nvcc_compile(&file, &suffix, flag)?)); 
    }
    for (name, mut ch) in children_proc
    { if !ch.wait().unwrap().success() { panic!("compiler process for output file {name:?} failed") } }
    println!("rerun-if-changed=cu-src/*");
    Ok(())
}