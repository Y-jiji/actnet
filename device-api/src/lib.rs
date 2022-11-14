use std::fmt::*;

#[derive(Debug)]
pub enum DevFunc<DevBox: Debug> {
    Add {read: (DevBox, DevBox), write: DevBox, meta: ()},
    Sub {read: (DevBox, DevBox), write: DevBox, meta: ()},
    Mul {read: (DevBox, DevBox), write: DevBox, meta: ()},
    Div {read: (DevBox, DevBox), write: DevBox, meta: ()},
    Cpy {read: DevBox, write: DevBox, meta: ()},
}

pub trait Device
where Self::DevBox: Debug {
    /// device box in device
    type DevBox;

    /// data buffer on host
    type DatBuf;

    /// launch a device function
    fn launch(&self, op: DevFunc<Self::DevBox>)
    { todo!("launch({op:?})") }

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