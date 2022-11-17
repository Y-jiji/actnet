use device_api::*;

mod mem;
mod raw;
mod dev;

struct CudaStream;

type Void = std::ffi::c_void;

#[derive(Debug)]
struct DatBuf {
    p: *mut Void,
    s: usize,
    t: Type,
}

impl Device for CudaStream {
    type DatBuf = DatBuf;
    type DevBox = DatBuf;
    type DevErr = raw::drv::cudaError_enum;
}