use crate::Void;
use std::collections::*;
use std::ptr::null_mut;

/// a memory management 'simulator'
trait MemState {
    fn new(lbound: *mut Void, rbound: *mut Void) -> Self;
    fn free(&mut self, ptr: *mut Void) -> Result<(), MemErr>;
    fn alloc(&mut self, size: usize) -> Result<*mut Void, MemErr>;
}

/// memory error
enum MemErr {
    OutOfMemory,
    InvalidPtr,
}

mod simple_state;
pub use simple_state::*;