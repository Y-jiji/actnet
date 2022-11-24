use crate::{NDArray, Device, Func, ComErr, DType, ManuallyDrop};

pub trait Rand<'a, D: Device>
where Self: Sized {
    fn rand_f32(shape: &[usize], device: &'a D) -> Result<Self, (ComErr, D::DevErr)>;
}

impl<'a, D: Device> Rand<'a, D> for NDArray<'a, D> {
    fn rand_f32(shape: &[usize], device: &'a D) -> Result<Self, (ComErr, D::DevErr)> {
        let symbol = device.emit(Func::RandF32 { read: (), meta: (shape.iter().product(),) })?
            .into_iter().next().unwrap();
        let shape = shape.to_vec();
        Ok(NDArray { symbol: ManuallyDrop::new(symbol), device, shape, dtype: DType::F32 })
    }
}


#[cfg(test)]
mod check_rand {
    use crate::{ops::Rand, NDArray};
    use device_toy::*;

    #[test]
    fn rand_f32_and_display() {
        let toy = Toy;
        let a: NDArray<Toy> = Rand::rand_f32(&[5, 12, 13], &toy).unwrap();
        println!("{a}");
    }
}