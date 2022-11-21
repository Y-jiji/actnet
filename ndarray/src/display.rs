use device_api::ArrayPrint;

use crate::{NDArray, Device};
use std::fmt::Display;

impl<D: Device> Display for NDArray<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let datbuf = self.device
            .seebox(self.devbox.clone());
        let datbuf = match datbuf {
            Err(_) => Err(std::fmt::Error)?,
            Ok(buf) => buf,
        };
        write!(f, "{}", datbuf.print(self.shape.clone()))
    }
}

#[cfg(test)]
mod check_display {
    use device_toy::*;

    fn display() {
        todo!("test displaying ndarray with toy");
    }
}