use std::fmt::Debug;
use std::result::*;

mod ops;
mod ext;
mod log;
mod typ;
mod mrg;
mod err;

pub use ops::*;
pub use ext::*;
pub use log::*;
pub use typ::*;
pub use mrg::*;
pub use err::*;

/// a device should be an internally mutable type
/// 
/// there are many symbols on a device. you can launch functions that read from and write to symbols. 
/// 
/// 5 associated functions and 3 associated types
/// 
/// for some applications, you guarantee every symbol to be written only once
/// 
/// on error, device returns a tuple of common error and device specific error
pub trait Device
where Self::Symbol: Debug + Symbol + Default + Eq, 
      Self::DatBox: Debug + DatBox + Default, 
      Self::DevErr: Debug + Default, 
      Self: Debug + Sized + PartialEq + Eq, {

    /// symbol on device, models a flat vector of given data type
    /// 
    /// symbol type is immutable and should not implement clone publicly (however, implementing clone to keep track of unexecuted device functions is a good practice)
    /// 
    /// default symbol should be empty i.e. of size 0
    type Symbol;

    /// data buffer on host, a unique reference with ownership like box type
    type DatBox;

    /// device specific error like cuda error or other
    type DevErr;

    /// emit a function to this device, i.e. push a function to execution queue
    fn emit(&self, func: Func<Self::Symbol>) -> Result<(), DevErr<Self>>
    { todo!("Device.emit({func:?})") }

    /// define a symbol on this device, msz: memory size **in bytes**, ty: datatype
    fn defn(&self, msz: usize, ty: DType) -> Result<Self::Symbol, DevErr<Self>>
    { todo!("Device.defn(msz:{msz:?}, ty:{ty:?})") }

    /// dump given symbol to a datbox, not consuming this symbol
    fn dump(&self, symbol: &Self::Symbol) -> Result<Self::DatBox, DevErr<Self>>
    { todo!("Device.dump({symbol:?})") }

    /// load given data to a new symbol
    fn load(&self, datbox: Self::DatBox, symbol: &mut Self::Symbol) -> Result<(), DevErr<Self>>
    { todo!("Device.load({datbox:?}, {symbol:?})") }

    /// drop a symbol without retrieving content
    fn drop(&self, symbol: Self::Symbol) -> Result<(), DevErr<Self>>
    { todo!("Device.drop({symbol:?})") }

    /// print the name for this device
    fn name(&self) -> String
    { todo!("Device.name()") }
}