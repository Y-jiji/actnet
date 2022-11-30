use crate::*;

pub trait Dump<D: Device> {
    fn dump(self) -> Result<D::DatBox, (ComErr, D::DevErr)>;
}

impl<'a, D: Device> Dump<D> for NDArray<'a, D> {
    fn dump(mut self) -> Result<D::DatBox, (ComErr, D::DevErr)> {
        // crazy...
        let r = self.device.dump(ManuallyDrop::into_inner(std::mem::take(&mut self.symbol)));
        std::mem::forget(self); r
    }
}

#[cfg(test)]
mod check_dump {
    use device_toy::Toy;
    use super::*;

    #[test]
    fn dump() {
        let toy = Toy;
        let a = NDArray::rand_f32(&[10, 5], &toy).unwrap();
        println!("{:?}", a.dump());
    }
}