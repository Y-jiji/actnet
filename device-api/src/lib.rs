use std::fmt::Debug;
use std::result::*;

mod ops;
mod ext;
mod log;

pub use ops::*;
pub use ext::*;
pub use log::*;

pub enum Either<A, B> {A(A), B(B)}

/// common error format
#[derive(Debug, Clone, Copy)]
pub enum ComErr {
    /// (wanted, total)
    MemNotEnough(usize, usize),
    /// invalid access
    MemInvalidAccess,
    /// function read from corrupted symbol (unintialized or written by failed functions)
    FuncReadCorrupted, 
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

/// a device should be an internally mutable type
/// 
/// there are many symbols on a device. you can launch functions that read from and write to symbols. 
/// 
/// for some applications, you guarantee every symbol to be written only once
pub trait Device
where Self::Symbol: Debug + Eq + DTyped + Default, 
      Self::DatBox: Debug + ArrayPrint + From<Vec<f32>> + From<Vec<f64>> + From<Vec<i32>> + From<Vec<i64>> + From<Vec<u8>> + Into<Vec<u8>>, 
      Self::DevErr: Debug + Default, 
      Self: Debug + Clone + Sized, {

    /// symbol on device, models a flat vector of given data type
    /// 
    /// symbol type is immutable and should not implement Clone
    /// 
    /// default symbol should be empty i.e. of size 0
    type Symbol;

    /// data buffer on host, a unique reference with ownership like box type
    type DatBox;

    /// device specific error like cuda error or other
    type DevErr;

    /// emit a function to this device, i.e. push a function to execution queue
    fn emit(&self, func: Func<Self::Symbol>) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("emit({func:?})") }

    /// define a symbol on this device
    fn defn(&self, size: usize, ty: DType) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("defn(size:{size:?}, ty:{ty:?})") }

    /// dump given symbol to a datbox, not consuming this symbol
    fn dump(&self, symbol: &Self::Symbol) -> Result<Self::DatBox, (ComErr, Self::DevErr)>
    { todo!("dupl({symbol:?})") }

    /// load given data to a new symbol
    fn load(&self, datbox: Self::DatBox, symbol: &mut Self::Symbol) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("load({datbox:?}, {symbol:?})") }

    /// drop a symbol without retrieving content
    fn drop(&self, symbol: Self::Symbol) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("drop({symbol:?})") }
}