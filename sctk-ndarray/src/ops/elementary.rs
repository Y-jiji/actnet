use crate::*;
use std::ops::{Add, Sub, Mul, Div};

impl<'a, D: Device> Add for &NDArray<'a, D> {
    type Output = Result<NDArray<'a, D>, (ComErr, D::DevErr)>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape { Err((ComErr::FuncInvalidInputMeta, D::DevErr::default()))? }
        let len = self.shape.iter().product();
        let sym = self.device.emit(Func::AddF32 { read: (&self.symbol, &rhs.symbol), meta: (len, ) })?;
        Ok(NDArray {
            symbol: ManuallyDrop::new(sym.into_iter().next().unwrap()),
            device: self.device,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
}

impl<'a, D: Device> Sub for &NDArray<'a, D> {
    type Output = Result<NDArray<'a, D>, (ComErr, D::DevErr)>;
    fn sub(self, rhs: Self) -> Self::Output {
        let len = self.shape.iter().product();
        let sym = self.device.emit(Func::SubF32 { read: (&self.symbol, &rhs.symbol), meta: (len, ) })?;
        Ok(NDArray {
            symbol: ManuallyDrop::new(sym.into_iter().next().unwrap()),
            device: self.device,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
}

impl<'a, D: Device> Mul for &NDArray<'a, D> {
    type Output = Result<NDArray<'a, D>, (ComErr, D::DevErr)>;
    fn mul(self, rhs: Self) -> Self::Output {
        let len = self.shape.iter().product();
        let sym = self.device.emit(Func::MulF32 { read: (&self.symbol, &rhs.symbol), meta: (len, ) })?;
        Ok(NDArray {
            symbol: ManuallyDrop::new(sym.into_iter().next().unwrap()),
            device: self.device,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
}

impl<'a, D: Device> Div for &NDArray<'a, D> {
    type Output = Result<NDArray<'a, D>, (ComErr, D::DevErr)>;
    fn div(self, rhs: Self) -> Self::Output {
        let len = self.shape.iter().product();
        let sym = self.device.emit(Func::DivF32 { read: (&self.symbol, &rhs.symbol), meta: (len, ) })?;
        Ok(NDArray {
            symbol: ManuallyDrop::new(sym.into_iter().next().unwrap()),
            device: self.device,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod check_elementary {
    use super::*;
    use device_toy::Toy;

    #[test]
    fn add_sub_mul_div() {
        let toy = Toy;
        let a = NDArray::rand_f32(&[5, 4, 3], &toy).unwrap();
        let b = NDArray::rand_f32(&[5, 4, 3], &toy).unwrap();
        let c = (&a + &b).unwrap();
        println!("array c:\n {c}");
        let c = (&a - &b).unwrap();
        println!("array c:\n {c}");
        let c = (&a * &b).unwrap();
        println!("array c:\n {c}");
        let c = (&a / &b).unwrap();
        println!("array c:\n {c}");
    }
}