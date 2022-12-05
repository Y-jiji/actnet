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
