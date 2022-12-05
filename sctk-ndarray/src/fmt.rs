use crate::{NDArray, Device};
use dvapi::*;
use std::fmt::*;

impl<'a,  D: Device, T: DevVal> Display for NDArray<'a, D, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self.dv.dump(&self.sy) {
            Err(_) => Err(Error)?,
            Ok(datbox) => Ok(write!(f, "shape:\n{:?}\nvalue:\n{}\n", self.sh, datbox.print(self.sh.clone()))?),
        }
    }
}

impl<'a, D: Device, T: DevVal> Debug 
for NDArray<'a, D, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "NDArray<{:?}, {:?}, {:?}>{{{:?}}}", self.ty, self.dv.name(), self.sh, self.sy)
    }
}