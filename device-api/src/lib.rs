use std::fmt::Debug;
use std::result::*;

mod ops;
pub use ops::*;

/// common error format
#[derive(Debug, Clone, Copy)]
pub enum ComErr {
    /// (wanted, total)
    MemNotEnough(usize, usize),
    /// invalid access
    MemInvalidAccess,
    /// invalid input length
    FuncInvalidInputLength,
    /// invalid input meta
    FuncInvalidInputMeta,
    /// invalid input type
    FuncInvalidInputType,
    /// function not implemented
    FuncNotimplemented,
    /// device initialization failed
    InitFailure,
}

pub trait ArrayPrint {
    fn print(&self, shape: Vec<usize>) -> String 
    { todo!("ArrayPrint::print({shape:?})"); }
}

/// a device is an internally mutable type
pub trait Device
where Self::Symbol: Debug + Eq + DTyped,
      Self::DatBox: Debug + ArrayPrint + From<Vec<f32>> + From<Vec<f64>> + From<Vec<i32>> + From<Vec<i64>>, {

    /// symbol on device, models a flat vector of given data type
    /// 
    /// symbol type is immutable
    type Symbol;

    /// data buffer on host, a unique reference with ownership like Box
    type DatBox;

    /// device specific error
    type DevErr;

    /// emit a function to this device
    fn emit(&self, func: Func<Self::Symbol>) -> Result<Vec<Self::Symbol>, (ComErr, Self::DevErr)>
    { todo!("launch({func:?})") }

    /// drop a symbol without retrieving content
    fn drop(&self, symbol: Self::Symbol) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("delsym({symbol:?})") }

    /// dump data from given symbol
    fn dump(&self, symbol: Self::Symbol) -> Result<Self::DatBox, (ComErr, Self::DevErr)>
    { todo!("dump({symbol:?})") }

    /// load given data to a new symbol 
    fn load(&self, datbox: Self::DatBox) -> Result<Self::Symbol, (ComErr, Self::DevErr)>
    { todo!("load({datbox:?})") }
}