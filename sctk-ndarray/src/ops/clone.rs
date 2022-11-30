use crate::*;

trait TryClone<'a, D: Device> {
    fn try_clone(&self) -> Result<NDArray<'a, D>, (ComErr, D::DevErr)>;
}

impl<'a, D: Device> TryClone<'a, D> for NDArray<'a, D> {
    fn try_clone(&self) -> Result<NDArray<'a, D>, (ComErr, D::DevErr)> {
        let symbol = self.device
            .emit(Func::Clone { read: (&self.symbol, ), meta: () })?
            .into_iter().next().unwrap();
        Ok(NDArray { symbol: ManuallyDrop::new(symbol), device: self.device, shape: self.shape.clone(), dtype: self.dtype })
    }
}

#[cfg(test)]
mod check_clone {
    use super::*;
    use device_toy::Toy;

    #[test]
    fn clone() {
        let toy = Toy;
        let a = NDArray::rand_f32(&[1, 12], &toy).unwrap();
        println!("{a}");
        let b = a.try_clone().unwrap();
        println!("{b}");
    }
}