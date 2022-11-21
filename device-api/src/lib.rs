use std::fmt::Debug;
use std::result::*;

mod ops;
pub use ops::*;

#[derive(Debug, Clone, Copy)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// common error format
#[derive(Debug, Clone, Copy)]
pub enum ComErr {
    /// (wanted, total)
    MemNotEnough(usize, usize),
    /// invalid access
    MemInvalidAccess,
    /// operation not implemented
    OpNoImpl,
    /// device initialization failed
    InitFailure,
}

pub trait ArrayPrint {
    fn print(&self, shape: Vec<usize>) -> String 
    { todo!("ArrayPrint::print({shape:?})"); }
}

pub trait Device
where Self::DevBox: Debug + Clone,
      Self::DatBuf: Debug + ArrayPrint {

    /// device box in device, serves as a mutable reference like RefCell
    type DevBox;

    /// data buffer on host, serves as a unique reference like Box
    type DatBuf;

    /// device error
    type DevErr;

    /// launch a device function
    fn launch(&self, func: DevFunc<Self::DevBox>) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("launch({func:?})") }

    /// allocate a new box on this device with filling data
    fn newbox(&self, datbuf: Self::DatBuf) -> Result<Self::DevBox, (ComErr, Self::DevErr)>
    { todo!("newbox({datbuf:?})") }

    /// delete a box
    fn delbox(&self, devbox: Self::DevBox) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("delbox({devbox:?})") }

    /// inspect devbox (dump bytes into a data buffer)
    fn seebox(&self, devbox: Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)>
    { todo!("seebox({devbox:?})") }
}