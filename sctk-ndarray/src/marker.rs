use crate::*;
use std::mem::*;

pub(crate) use std::marker::PhantomData;
pub(crate) type NoDrop<T> = ManuallyDrop<T>;
pub(crate) fn nodrop<T>(x: T) -> NoDrop<T> { NoDrop::new(x) }
pub(crate) fn phant<T>() -> PhantomData<T> { PhantomData }

pub trait Num {}

impl Num for f32 {}
impl Num for f64 {}
impl Num for i32 {}
impl Num for i64 {}

pub trait Basic {
    /// give correspondent data type
    fn ty() -> DType;
    /// wrap self with data type
    fn wrap(self) -> WrapVal;
    /// memory size * x
    fn msize(x: usize) -> usize;
}

macro_rules! impl_basic {
    ($LowerCase: tt, $BigCase: ident) => {
        impl Basic for $LowerCase {
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

impl_basic!(f32, F32);
impl_basic!(f64, F64);
impl_basic!(i32, I32);
impl_basic!(i64, I64);
impl_basic!(bool, Bool);