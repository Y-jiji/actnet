use dvapi::*;
use std::mem::take;

mod fmt;
mod ops;
mod cast;
mod marker;

pub use fmt::*;
pub use ops::*;
pub use cast::*;
use marker::*;

/// n-dimensional array
pub struct NDArray<'a, D: Device, T: Sized+'static> {
    /// the correspondent symbol, lifetime ended by device.drop(symbol)
    sy: NoDrop<D::Symbol>,
    /// data type
    ty: PhantomData<T>,
    /// the correspondent device
    dv: &'a D,
    /// shape of an ndarray
    sh: Vec<usize>,
    /// flatten length of an ndarray
    ln: usize,
}

impl<'a, D: Device, T: Sized+'static> NDArray<'a, D, T> {
    /// length
    pub fn len(&self) -> usize {self.ln}
    /// shape
    pub fn shape(&self) -> &[usize] {&self.sh}
    /// failure tolerant drop, returns self when drop fails
    pub fn drop(mut self) -> Result<(), (Self, ComErr, D::DevErr)> {
        match self.dv.drop(take(&mut self.sy)) {
            Ok(()) => Ok(()),
            Err((comerr, deverr)) => Err((self, comerr, deverr))
        }
    }
    /// get stuck until drop succeed or device panic
    /// 
    /// unsafe note(!) 
    /// infinite loop is possible if device cannot recover from previous errors
    pub unsafe fn drop_retry(mut self) {loop{
        match self.dv.drop(take(&mut self.sy)) {
            Ok(()) => break,
            Err(_) => continue,
        }
    }}
}

impl<'a, D: Device, T: Sized+'static> Drop for NDArray<'a, D, T> {
    fn drop(&mut self) {
        // panic-on-error drop
        let _r = self.dv.drop(take(&mut self.sy));
        debug_assert!(_r.is_ok(), "{_r:?}");
    }
}