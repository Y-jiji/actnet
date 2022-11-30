use dvapi::*;

mod display;
pub use display::*;

mod death;
pub use death::*;

mod ops;
pub use ops::*;

use std::mem::*;

#[derive(Debug)]
pub struct NDArray<'a, D: Device> {
    /// the correspondent symbol, lifetime ended by device.drop(symbol)
    symbol: ManuallyDrop<D::Symbol>,
    /// the correspondent device
    device: &'a D,
    /// shape of an ndarray
    shape: Vec<usize>,
    /// data type
    dtype: DType,
}

impl<'a, D: Device> DTyped for NDArray<'a, D> {
    fn dtype(&self) -> DType {self.dtype}
}