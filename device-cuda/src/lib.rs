 use device_api::*;

mod raw;
mod mem;
mod dev;

struct CudaStream;

type Void = std::ffi::c_void;

#[derive(Debug)]
struct DevBox {
    /// base pointer
    p: *mut Void,
    /// length of the boxed array
    s: usize,
    /// type of the boxed array
    t: Type,
}

impl Device for CudaStream {
    type DatBuf = DevBox;
    type DevBox = DevBox;
    type DevErr = raw::drv::cudaError_enum;
}