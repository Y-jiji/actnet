use std::ptr::null_mut;

use crate::*;

#[derive(Debug, PartialEq, Eq)]
pub struct Symbol {
    /// data type
    pub(crate) dtype: DType,
    /// inner pointer
    pub(crate) inner: *mut u8,
    /// memory size
    pub(crate) msize: usize,
}

impl Default for Symbol {
    fn default() -> Self {
        Symbol { dtype: DType::Bool, inner: null_mut(), msize: 0 }
    }
}

impl Symbol {
    pub(crate) fn ptr<T>(&self) -> *mut T {
        self.inner as *mut T
    }
}

impl DTyped for Symbol {
    fn dtype(&self) -> DType {self.dtype}
}

impl Drop for Symbol {
    fn drop(&mut self) {
        panic!("symbol drop should be managed by device. please use Toy::drop(...)")
    }
}