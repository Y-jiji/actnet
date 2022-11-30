use dvapi::*;
use mem_allocator::*;

mod cuda_wrap;
mod cuda_stream;
mod cuda_device;
mod cuda_bundle;

type Void = std::ffi::c_void;

#[derive(Debug)]
struct DevBox {
    /// base pointer
    p: *mut Void,
    /// length of the boxed array
    s: usize,
    /// type of the boxed array
    t: DType,
}

// impl Device for CudaStream {
//     type DatBuf = DevBox;
//     type DevBox = DevBox;
//     type DevErr = raw::drv::cudaError_enum;
// }