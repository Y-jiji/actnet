use device_api::*;

mod display;
use display::*;

mod ops;
use ops::*;

struct NDArray<'a: 'b, 'b, D: Device<'a, 'b>> {
    /// the correspondent symbol
    symbol: D::Symbol,
    /// the correspondent device
    device: &'b D,
    /// shape of an ndarray
    shape: Vec<usize>,
    /// data type
    dtype: DType,
}