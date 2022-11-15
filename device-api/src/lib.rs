use std::fmt::Debug;
use std::result::*;

#[derive(Debug)]
pub enum Type {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// common error format
#[derive(Debug)]
pub enum ComErr {

}

#[derive(Debug)]
pub enum DevFunc<DevBox: Debug> {
    AddF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    SubF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    MulF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    DivF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    ContractF32{read: (DevBox, DevBox), write: DevBox, meta: (usize, usize, usize)},
    Cpy {read: DevBox, write: DevBox, meta: ()},
}

pub trait Device
where Self::DevBox: Debug {
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
    fn newbox(&self, size: usize) -> Result<Self::DevBox, (ComErr, Self::DevErr)>
    { todo!("newbox({size:?})") }

    /// delete a box and dump bytes into a data buffer
    fn delbox(&self, devbox: Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)>
    { todo!("delbox({devbox:?})") }

    /// inspect devbox (dump bytes into a data buffer)
    fn seebox(&self, devbox: &Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)>
    { todo!("seebox({devbox:?})") }
}