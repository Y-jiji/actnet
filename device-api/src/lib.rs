use std::fmt::Debug;
use std::result::*;

mod ops;
pub use ops::*;

#[derive(Debug)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// common error format
#[derive(Debug)]
pub enum ComErr {
    /// (wanted, total)
    OutOfMemory(usize, usize),
    /// operation not implemented
    OpNoImpl,
}

pub trait ArrayPrint {
    fn print(&self, shape: Vec<usize>) -> String 
    { todo!("ArrayPrint::print({shape:?})"); }
}

pub trait Device
where Self::DevBox: Debug,
      Self::DatBuf: ArrayPrint {

    /// device box in device
    type DevBox;

    /// data buffer on host
    type DatBuf;

    /// device error
    type DevErr;

    /// launch a device function
    fn launch(&self, func: DevFunc<Self::DevBox>) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("launch({func:?})") }

    /// allocate a new box on this device
    fn newbox(&self, size: usize, dtype: DType) -> Result<Self::DevBox, (ComErr, Self::DevErr)>
    { todo!("newbox({size:?}, {dtype:?})") }

    /// delete a box
    fn delbox(&self, devbox: Self::DevBox) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("delbox({devbox:?})") }

    /// inspect devbox (dump bytes into a data buffer)
    fn seebox(&self, devbox: Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)>
    { todo!("seebox({devbox:?})") }
}