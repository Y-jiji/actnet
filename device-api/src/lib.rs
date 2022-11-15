use std::fmt::*;

#[derive(Debug)]
pub enum DevFunc<DevBox: Debug> {
    AddF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    SubF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    MulF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    DivF32 {read: (DevBox, DevBox), write: DevBox, meta: ()},
    Cpy {read: DevBox, write: DevBox, meta: ()},
}

pub trait Device
where Self::DevBox: Debug {
    /// device box in device
    type DevBox;

    /// data buffer on host
    type DatBuf;

    /// launch a device function
    fn launch(&self, func: DevFunc<Self::DevBox>)
    { todo!("launch({func:?})") }

    /// allocate a new box on this device
    fn newbox(&self, size: usize) -> Self::DevBox
    { todo!("newbox({size:?})") }

    /// delete a box and dump bytes into a data buffer
    fn delbox(&self, devbox: Self::DevBox) -> Self::DatBuf
    { todo!("delbox({devbox:?})") }

    /// inspect devbox (dump bytes into a data buffer)
    fn seebox(&self, devbox: &Self::DevBox) -> Self::DatBuf
    { todo!("seebox({devbox:?})") }
}