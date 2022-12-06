use std::process::{Command, Child};
use std::path::PathBuf;
use std::error::Error;

const NVCC: &str = "nvcc";
const CU_SRC_PATH: &str = "cu-src";
const CU_TRG_PATH: &str = "cu-target";

pub fn nvcc_compile(
    file: &str,
    args: Vec<&str>,
) -> Result<Child, Box<dyn Error>> {
    let child = Command::new(NVCC)
        .arg(PathBuf::new().join(CU_SRC_PATH).join(String::new() + file + ".cu"))
        .arg("-o")
        .arg(PathBuf::new().join(CU_TRG_PATH).join(String::new() + file + ".ptx"))
        .arg("--ptx")
        .arg("-arch=native")
        .args(args)
        .spawn()?;
    Ok(child)
}

pub fn detect_cufile() -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let path = PathBuf::new().join(CU_SRC_PATH);
    let mut result = Vec::new();
    for file in path.read_dir()? {
        let file = dbg!(file)?.file_name();
        let file = file.to_str().unwrap();
        if !file.ends_with(".cu") { continue; }
        let file = String::from(file.strip_suffix(".cu").unwrap());
        let flag = std::fs::read_to_string(path.display().to_string() + &file + ".flag");
        match flag {
            Ok(flag) => for flag in flag.split("\n") {
                result.push((file.clone(), flag.to_owned()));
            },
            Err(_) => result.push((file.clone(), String::new()))
        }
    }
    Ok(result)
}