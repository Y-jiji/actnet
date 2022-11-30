use dvapi::ArrayPrint;

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