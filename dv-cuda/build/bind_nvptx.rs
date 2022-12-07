use std::process::{Command, Child};
use std::path::PathBuf;
use std::error::Error;

const NVCC: &str = "nvcc";
const CU_SRC_PATH: &str = "cu-src";
const CU_TRG_PATH: &str = "cu-target";

pub fn nvcc_compile(
    file: &str,
    suffix: &str,
    flag: Vec<&str>,
) -> Result<Child, Box<dyn Error>> {
    let child = Command::new(NVCC)
        .arg(PathBuf::new().join(CU_SRC_PATH).join(String::new() + file + ".cu"))
        .args(flag)
        .arg("-O2")
        .arg("-o")
        .arg(dbg!(PathBuf::new().join(CU_TRG_PATH).join(String::new() + file + suffix + ".ptx")))
        .arg("--ptx")
        .arg("-arch=native")
        .spawn()?;
    Ok(child)
}

pub fn detect_cufile() -> Result<Vec<(String, String, String)>, Box<dyn Error>> {
    let path = PathBuf::new().join(CU_SRC_PATH);
    let mut result = Vec::new();
    for file in path.read_dir()? {
        let file = file?.file_name();
        let file = file.to_str().unwrap();
        if !file.ends_with(".cu") { continue; }
        let file = String::from(file.strip_suffix(".cu").unwrap());
        let desc = std::fs::read_to_string(path.join(file.clone() + ".flag"));
        let desc = match desc {
            Ok(desc) => desc,
            Err(_) => {result.push((file.clone(), String::new(), String::new())); continue;}
        };
        for line in desc.lines().map(|s| s.trim()) {
            if let Some((suffix, flags)) = line.split_once(char::is_whitespace) {
                result.push((file.clone(), suffix.to_owned(), flags.to_owned()))
            }
        }
    }
    Ok(result)
}