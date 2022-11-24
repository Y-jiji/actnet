use device_api::ArrayPrint;

use crate::{NDArray, Device, Func};
use std::fmt::Display;

impl<'a,  D: Device> Display for NDArray<'a, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let clone_result = self.device.emit(Func::Clone { read: (&self.symbol, ), meta: () });
        let b = match clone_result { Ok(ab) => ab, Err(_) => Err(std::fmt::Error)? }.into_iter().next().unwrap();
        match self.device.dump(b) {
            Err(_) => Err(std::fmt::Error)?,
            Ok(datbox) => Ok(write!(f, "{}", datbox.print(self.shape.clone()))?),
        }
    }
}

#[cfg(test)]
mod check_display {
    use crate::{ops::Rand, NDArray};
    use device_toy::*;

    #[test]
    fn display() {
        let toy = Toy;
        let a: NDArray<Toy> = Rand::rand_f32(&[5, 12, 13], &toy).unwrap();
        println!("{a}");
    }
}