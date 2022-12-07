use dvapi::*;
use mm_alloc::*;

mod tool;
// mod cuda_stream;
mod cuda_device;
mod cuda_bundle;

pub(crate) type Void = std::ffi::c_void;