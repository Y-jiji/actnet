use std::process::{Command, Child};
use std::path::PathBuf;
use std::error::Error;

const NVCC: &str = "nvcc";
const CU_SRC_PATH: &str = "cu-src";
const CU_TRG_PATH: &str = "cu-target";

pub fn nvcc_compile(
    file: &str
) -> Result<Child, Box<dyn Error>> {
    let child = Command::new(NVCC)
        .arg(PathBuf::new().join(CU_SRC_PATH).join(String::new() + file + ".cu"))
        .arg("-o")
        .arg(PathBuf::new().join(CU_TRG_PATH).join(String::new() + file + ".ptx"))
        .arg("--ptx")
        .arg("-arch=native")
        .spawn()?;
    Ok(child)
}

pub fn detect_cufile() -> Result<Vec<String>, Box<dyn Error>> {
    let path = PathBuf::new().join(CU_SRC_PATH);
    let mut result = Vec::new();
    for file in path.read_dir()? {
        result.push(dbg!(file)?.file_name().to_str()
            .unwrap().strip_suffix(".cu")
            .expect("file not ended with .cu").to_owned());
    }
    Ok(result)
}