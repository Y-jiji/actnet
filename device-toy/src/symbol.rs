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
        panic!("symbol drop should be managed by device. use toy.drop(...) instead")
    }
}