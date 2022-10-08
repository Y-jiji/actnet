use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::*;
use std::mem::*;
use crate::*;

pub type Void = std::ffi::c_void;

pub struct ToyBox {
    ptr: *mut Void,
    len: usize,
}

impl ToyBox {
    fn new(size: usize) -> ToyBox {
        let ptr = unsafe{alloc_zeroed(Layout::array::<u8>(size).unwrap())} as *mut _;
        let len = size;
        return ToyBox { ptr, len }
    }
}

impl Drop for ToyBox {
    fn drop(&mut self) {
        unsafe{dealloc(self.ptr as *mut _, Layout::array::<u8>(self.len).unwrap())};
    }
}

pub struct ToyUniStream;

impl ToyUniStream {
    fn add_f32(a: &ToyBox, b: &ToyBox, c: &ToyBox) {
        let aptr = a.ptr as *mut f32;
        let bptr = b.ptr as *mut f32;
        let cptr = c.ptr as *mut f32;
        let size = a.len / size_of::<f32>();
        for i in 0..size {
            unsafe{*cptr.add(i) = *aptr.add(i) + *bptr.add(i)}
        }
    }
}

impl Device for ToyUniStream {
    type DevBox = ToyBox;
    type DevErr = String;
    fn send(msg: DevMsg<Self>) -> Result<(), Self::DevErr> {
        match msg {
            DevMsg::NewBox { size, dst } =>
                unsafe{*dst = ToyBox::new(size)},
            DevMsg::PutBox { src, dst } => 
                unsafe{copy(src as *mut u8, dst.ptr as *mut u8, dst.len)}
            DevMsg::GetBox { src, dst } => 
                unsafe{copy(src.ptr as *mut u8, dst as *mut u8, src.len)}
            DevMsg::DelBox { src } =>
                drop(src),
            DevMsg::Launch { function, src, dst, meta_u, meta_f } => {
                if function == "add_f32" {
                    if src[0].len != src[1].len || src[1].len == dst.len
                    { return Err("size doesn't match".to_string()); }
                    Self::add_f32(&src[0], &src[1], &dst)
                }
            },
            DevMsg::CpyBox { src, dst } => {
                if src.len != dst.len { return Err("size doesn't match".to_string()) }
                unsafe{copy(src.ptr, dst.ptr, src.len)}
            },
            DevMsg::HookUp { src } => {
                (src)();
            }
        };
        Ok(())
    }
}