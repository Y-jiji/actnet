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

impl DTyped for ExtSymbol {fn dtype(&self) -> DType {self.t}}

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
    fn ext_init(&self, size: usize) -> Result<(usize, Vec<usize>), (ComErr, Self::DevErr)>
    { todo!("{size:?}") }

    /// unload the extension indicated by id, release its resources
    fn ext_kill(&self, id: usize) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("{id:?}") }

    /// emit a copy function to copy from one memory segment to another
    /// 
    /// id: id of this extension; 
    /// src: (segment_number, offset); 
    /// dst: (segment_number, offset); 
    /// size: the size to copy in byte; 
    fn ext_copy(&self, id: usize, 
        src: &ExtSymbol,
        dst: &mut ExtSymbol
    ) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("copy({id:?}, {src:?}, {dst:?})") }

    /// emit a device function on leaf device (physical device with a unified memory address)
    fn ext_emit(&self, id: usize, func: Func<ExtSymbol>) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("exec({id:?}, {func:?})") }
}