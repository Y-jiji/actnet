extern crate bindgen;
use std::env::current_dir;
use std::error::Error; 
use std::path::*;
use std::fs::*;

const CUDA_INCLUDE_PATH: &str = env!(
    "CUDA_INCLUDE_PATH", 
    "In build script: Environment variable CUDA_INCLUDE_PATH is missing. It should be where \"cuda.h\" file lives"
);

const CUDA_LIBRARY_PATH: &str = env!(
    "CUDA_LIBRARY_PATH",
    "In build script: Environment variable CUDA_LIBRARY_PATH is missing. It should be where \"cuda.lib\" or \"libcuda.a\" file lives"
);

const CUDA_BINDING_PATH: &str = "cu-bind";

pub fn bind_cuda(
    include: &str, 
    writeto: &str,
    library: &str,
) -> Result<(), Box<dyn Error>> {
    let cd = current_dir()?;
    let include_path = PathBuf::new().join(CUDA_INCLUDE_PATH).join(include);
    let writeto_path = PathBuf::new().join(cd).join(CUDA_BINDING_PATH).join(writeto);
    // let library_path = PathBuf::new().join(CUDA_LIBRARY_PATH).join(library);
    let mut builder = bindgen::Builder::default();
    builder = builder
        .clang_args(["-I", CUDA_INCLUDE_PATH]);
    println!("cargo:rustc-link-search={CUDA_LIBRARY_PATH}");
    println!("cargo:rustc-link-lib={library}");
    builder = builder
        .header(include_path.to_str().unwrap())
        .rustified_enum(".*enum");
    let binding = builder.generate()?;
    assert!(writeto_path.exists() && writeto_path.is_file(), "There is no file {writeto_path:?}");
    let writeto_file = Box::new(File::create(&writeto_path).expect(&format!("Cannot open file {writeto_path:?}")));
    binding.write(writeto_file).expect(&format!("Cannot write to {writeto_path:?}"));
    Ok(())
}