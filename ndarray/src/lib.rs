use device_api::*;

mod display;
use display::*;

mod ops;
use ops::*;

pub struct NDArray<'a, D: Device> {
    /// the correspondent symbol
    symbol: D::Symbol,
    /// the correspondent device
    device: &'a D,
    /// shape of an ndarray
    shape: Vec<usize>,
    /// data type
    dtype: DType,
}