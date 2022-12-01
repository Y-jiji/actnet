//! this module defines essential traits that a device extension to operate the device
 
use crate::*;

/// extensions implement memory allocation and releasing on this type
/// 
/// because this type serves as point on device, you should never implement 'shallow clone' on it
#[derive(Debug)]
pub struct ExtSymbol {
    /// segment number
    pub n: usize,
    /// offset
    pub o: usize,
    /// size in bytes
    pub s: usize,
    /// data type 
    pub t: DType,
}

impl Symbol for ExtSymbol {
    fn dtype(&self) -> DType {self.t}
    fn msize(&self) -> usize {self.s}
}

/// with some known features about system load, customized memory management over devices can be critical to system performance. 
/// 
/// this trait provides an api for these invasive memory management policies
pub trait DevExt
where Self: Device {
    /// initialize some pieces of memory for a device extension to manage
    /// 
    /// together with any extension id
    /// 
    /// (offset, a mapping indicated by vec : segment_number -> size)
    fn ext_defn(&self, size: usize) -> Result<(usize, Vec<usize>), DevErr<Self>>
    { todo!("{size:?}") }

    /// unload the extension indicated by id, release its resources to device
    fn ext_drop(&self, id: usize) -> Result<(), DevErr<Self>>
    { todo!("{id:?}") }

    /// emit a device function on leaf device (physical device with a unified memory address)
    fn ext_emit(&self, id: usize, func: Func<ExtSymbol>) -> Result<(), DevErr<Self>>
    { todo!("exec({id:?}, {func:?})") }
}