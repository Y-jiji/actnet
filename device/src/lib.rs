use std::pin::*;

pub type Void = std::ffi::c_void;

pub trait Device {
    type DevBox;
    type DevErr;
    fn send(msg: DevMsg<Self>) -> Result<(), Self::DevErr>;
}

pub enum DevMsg<D: Device + ?Sized> {
    /// register a new box (value returned to *dst), this is an immediate function on host
    NewBox {size: usize, dst: *mut D::DevBox},
    /// launch a job that read from source box and write to destination box
    Launch {function: String, 
        src: Vec<D::DevBox>, dst: D::DevBox, 
        meta_u: Vec<usize>,  meta_f: Vec<f32>},
    /// launch a job that delete box
    DelBox {src: D::DevBox},
    /// launch a job that copy data from source box to destination box
    CpyBox {src: D::DevBox, dst: D::DevBox},
    /// launch a job that put data to destination box
    PutBox {src: *mut Void, dst: D::DevBox},
    /// launch a job that get data from source box
    GetBox {src: D::DevBox, dst: *mut Void},
    /// launch a function after previous launched jobs finished
    HookUp {src: Box<dyn FnOnce() + Send>},
}

pub mod cuda_local_uni;
pub mod cuda_physic;

mod toy_uni;
pub use toy_uni::*;