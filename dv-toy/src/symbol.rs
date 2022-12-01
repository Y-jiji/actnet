use std::ptr::null_mut;

use crate::*;

#[derive(Debug, PartialEq, Eq)]
pub struct ToySymbol {
    /// data type
    pub(crate) dtype: DType,
    /// inner pointer
    pub(crate) inner: *mut u8,
    /// memory size
    pub(crate) msize: usize,
}

impl Default for ToySymbol {
    fn default() -> Self {
        ToySymbol { dtype: DType::Bool, inner: null_mut(), msize: 0 }
    }
}

impl ToySymbol {
    pub(crate) fn ptr<T>(&self) -> *mut T {
        self.inner as *mut T
    }
}

impl Drop for ToySymbol {
    fn drop(&mut self) {
        panic!("symbol drop should be managed by device. please use Toy::drop(...)")
    }
}

impl Symbol for ToySymbol {
    fn dtype(&self) -> DType {self.dtype}
    fn msize(&self) -> usize {self.msize}
}