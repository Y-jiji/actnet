//! traits for associated types in device

/* ------------------------------------------------------------------------------------------ */
/*                                         data types                                         */
/* ------------------------------------------------------------------------------------------ */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32, 
    F64, 
    I32,
    I64, 
    Bool,
    FallBack
}

pub use DType::{
    F32 as DF32, F64 as DF64, 
    I32 as DI32, I64 as DI64, 
    Bool as DBool, 
    FallBack as DFallBack
};

#[derive(Debug)]
pub enum WrapVal {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
    FallBack,
}

#[derive(Debug)]
pub enum WrapVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    FallBack,
}

pub use WrapVec::{
    F32 as WF32, F64 as WF64,
    I32 as WI32, I64 as WI64,
    Bool as WBool,
    FallBack as WFallback
};

/// ### *brief*
/// ```pseudocode
/// data type recognized by device. 
/// can be wrapped into wrapval type. 
/// ```
pub trait DevVal {
    /// give correspondent data type
    fn ty() -> DType;
    /// wrap self with data type
    fn wrap(self) -> WrapVal;
    /// memory size from array size
    fn msize(len: usize) -> usize;
}

use std::mem::size_of;

macro_rules! impl_from_val {
    ($LowerCase: tt, $BigCase: ident) => {
        impl DevVal for $LowerCase {
            #[inline]
            fn ty() -> DType {DType::$BigCase}
            #[inline]
            fn wrap(self) -> WrapVal {WrapVal::$BigCase(self)}
            #[inline]
            fn msize(x: usize) -> usize {
                if Self::ty() == DBool { return (x + 7) & (usize::MAX<<3) }
                else { return x * size_of::<Self>() }
            }
        }
    };
}

impl_from_val!(f32, F32);
impl_from_val!(f64, F64);
impl_from_val!(i32, I32);
impl_from_val!(i64, I64);
impl_from_val!(bool, Bool);

/* ------------------------------------------------------------------------------------------ */
/*                        common behaviours for device associated types                       */
/* ------------------------------------------------------------------------------------------ */

pub trait Symbol {
    /// get data type
    fn dtype(&self) -> DType;
    /// get memory size
    fn msize(&self) -> usize;
}

pub trait DatBox 
where Self: Sized {
    /// get data type
    fn dtype(&self) -> DType;
    /// get memory size
    fn msize(&self) -> usize;
    fn print(&self, shape: Vec<usize>) -> String 
    { todo!("ArrayPrint::print({shape:?})"); }
    fn as_vec(self) -> WrapVec
    { todo!("as_vec()") }
    fn from_vec(x: WrapVec) -> Self
    { todo!("from_vec({x:?})") }
    fn as_byte(self) -> Vec<u8>
    { todo!("as_byte()") }
    fn from_byte(x: Vec<u8>, ty: DType) -> Self
    { todo!("from_byte({x:?}, {ty:?})") }
}