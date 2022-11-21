use crate::Void;
use device_api::*;
use mem_allocator::MemShadow;
use std::rc::Rc;
use std::cell::RefCell;
use super::stream::*;

#[derive(Debug)]
struct DevBoxInner {
    /// box number
    n: usize,
    /// pointer for fast access
    p: *mut Void,
    /// size for boundary checking
    s: usize, 
    /// memory manager
    mm: Rc<RefCell<MemShadow<8>>>,
}

impl Drop for DevBoxInner {
    fn drop(&mut self) {
        #[cfg(test)] println!("drop({self:?})");
        let is_ok = self.mm.borrow_mut().free(self.n).is_ok();
        debug_assert!(is_ok, "unexpected free({:?}) where n refers to nothing", self.n);
    }
}

#[derive(Debug)]
pub struct DevBox {inner: Rc<DevBoxInner>}

impl Clone for DevBox {
    fn clone(&self) -> Self {
        DevBox {inner: Rc::clone(&self.inner)}
    }
}

impl DevBox {
    /// get a new uninitialized device box
    pub fn new_uninit(device: &CudaStream, s: usize) -> Result<DevBox, ComErr> {
        let mm = Rc::clone(&device.mm);
        let n = mm.borrow_mut().alloc(s)?;
        let p = mm.borrow().getptr(n);
        let s = mm.borrow().getsiz(n);
        return Ok(DevBox { inner: Rc::new(DevBoxInner { n, p, mm, s }) });
    }
    /// get the inner pointer on device
    #[inline]
    pub fn as_ptr(&self) -> *mut Void { self.inner.p }
    /// get the size for boundary check
    #[inline]
    pub fn size(&self) -> usize { self.inner.s }
}

#[cfg(test)]
mod check_devbox {
    use super::*;

    #[test]
    fn new_and_drop() {
        // devbox should be a counted reference, so the referred memory should be only released once
        let device = CudaStream::new(1024).unwrap();
        println!("Initialization");
        let devbox_1 = DevBox::new_uninit(&device, 128).unwrap();
        let devbox_2 = devbox_1.clone();
        let devbox_3 = devbox_1.clone();
        drop(devbox_1);
        drop(devbox_3);
        drop(devbox_2);
    }
}