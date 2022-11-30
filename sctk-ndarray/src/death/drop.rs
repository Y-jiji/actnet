use crate::*;

impl<'a, D: Device> Drop for NDArray<'a, D> {
    fn drop(&mut self) {
        let r = self.device.drop(std::mem::take(&mut self.symbol));
        debug_assert!(r.is_ok(), "{r:?}");
    }
}