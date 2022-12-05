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

pub(crate)
const SCTK_CRATE_NAME: &str = "sctk-ndarray";

/// n-dimensional array
pub struct NDArray<'a, D: Device, T: DevVal> {
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
    /// nil means 'already dropped'
    nil: bool,
}

impl<'a, D: Device, T: DevVal> NDArray<'a, D, T> {

    /// ## *output*
    /// ```pseudocode
    /// length of flattened ndarray
    /// ```
    pub fn len(&self) -> usize {self.ln}

    /// ## *output*
    /// ```pseudocode
    /// n-element array, indicating shape of this ndarray
    /// ```
    pub fn shape(&self) -> &[usize] {&self.sh}

    /// ## *effect*
    /// ```pseudocode
    /// if drop succeed, mark itself as nil=true
    /// else, return device error
    /// ```
    /// ## *output*
    /// ```pseudocode
    /// if it cannot be dropped, return device error
    /// else, return ()
    /// ```
    /// ## *note*
    /// ```pseudocode
    /// for failure-tolerant application, we recommend using self.drop() instead of implicit drop, 
    /// since implicit drop panics on device error, even if the device is remote
    /// ```
    pub fn drop(&mut self) -> Result<(), DevErr<D>> {
        self.dv.drop(
            take(&mut self.sy)
        ).map(|()| { self.nil = true; () })
    }
}

impl<'a, D: Device, T: DevVal> Drop for NDArray<'a, D, T> {
    fn drop(&mut self) {
        if self.nil { return; }
        // panic-on-error drop
        let _r = self.dv.drop(take(&mut self.sy));
        debug_assert!(_r.is_ok(), "{_r:?}");
    }
}