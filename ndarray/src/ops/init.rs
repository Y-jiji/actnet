use crate::{NDArray, Device, Func, ComErr, DType};

pub trait Rand<'a: 'b, 'b, D: Device>
where Self: Sized {
    fn rand_f32(shape: &[usize], device: &D) -> Result<Self, (ComErr, D::DevErr)>;
}

impl<'a: 'b, 'b, D: Device<'a, 'b>> Rand<'a, 'b, D> for NDArray<'a, 'b, D> {
    fn rand_f32(shape: &[usize], device: &D) -> Result<Self, (ComErr, D::DevErr)> {
        let symbol = device.emit(Func::RandF32 { read: (), meta: (shape.iter().product(),) })?
            .into_iter().next().unwrap();
        let shape = shape.to_vec();
        Ok(NDArray { symbol, device, shape, dtype: DType::F32 })
    }
}