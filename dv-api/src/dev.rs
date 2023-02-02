use crate::exp::{ExpHandle, PlaceHolder};

pub trait Device
where
    Self: Sized,
    Self::DevPointer: DevPointer<Self>,
    Self::DevHandle: DevHandle<Self>,
{
    type DevPointer;
    type DevHandle;
    type DevError;
    type RefThis;
    fn compile(&mut self, handle: ExpHandle) -> Result<Self::DevHandle, Self::DevError>;
}

pub trait DevPointer<D>
where
    Self: Clone,
    Self::HostType: From<Self> + Into<Self>,
{
    type HostType;
    fn internalize(&self) -> PlaceHolder;
    fn load(data: &Self::HostType, device: &mut D) -> Self;
    fn dump(&self) -> Self::HostType;
}

pub trait DevHandle<D>
where
    Self: Clone,
    D: Device,
{
    fn internalize(&self) -> PlaceHolder;
    fn arg(self, input: D::DevPointer) -> Result<D::DevPointer, D::DevError>;
    fn result(self) -> Result<D::DevPointer, D::DevError>;
}