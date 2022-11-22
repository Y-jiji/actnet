use device_api::*;

mod display;
use display::*;

mod ops;
use ops::*;

struct NDArray<D: Device> {
    devbox: D::DevBox,
    /// 
    device: D,
    /// shape of an ndarray
    shape: Vec<usize>,
    /// data type
    dtype: DType,
}
