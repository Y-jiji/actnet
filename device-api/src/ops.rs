//! this module defines basic operations and operands for a device and correspondent data box

use std::fmt::Debug;
use crate::*;

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

pub use DType::F32 as DF32;
pub use DType::F64 as DF64;
pub use DType::I32 as DI32;
pub use DType::I64 as DI64;
pub use DType::Bool as DBool;
pub use DType::FallBack as DFallBack;

pub trait DTyped {
    fn dtype(&self) -> DType;
}

/* ------------------------------------------------------------------------------------------ */
/*                                        datbox traits                                       */
/* ------------------------------------------------------------------------------------------ */

pub trait ArrayPrint {
    fn print(&self, shape: Vec<usize>) -> String 
    { todo!("ArrayPrint::print({shape:?})"); }
}

pub trait AsBytes {
    fn as_bytes(self) -> Vec<u8>;
}

pub enum WrapVec {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    FallBack,
}

pub use WrapVec::F32 as WF32;
pub use WrapVec::F64 as WF64;
pub use WrapVec::I32 as WI32;
pub use WrapVec::I64 as WI64;
pub use WrapVec::Bool as WBool;
pub use WrapVec::FallBack as WFallback;

pub trait AsVec {
    fn as_vec(self) -> WrapVec;
}

/* ------------------------------------------------------------------------------------------ */
/*                                       operation types                                      */
/* ------------------------------------------------------------------------------------------ */

#[derive(Debug)]
/// device function on device, will get even larger in near future
/// (i: input Symbol;  o: output Symbol;  m: meta data)
pub enum Func<'t, Symbol: Debug + DTyped> {
    /// c(i) <-- a(i) + b(i)
    AddF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// (c, )
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// c(i) <-- a(i) - b(i)
    SubF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// (c, )
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// c(i) <-- a(i) * b(i)
    MulF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// (c, )
        o: (&'t mut Symbol, ),
        /// length of a, length of b
        m: (usize, )
    },
    /// c(i) <-- a(i) / b(i)
    DivF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// (c, )
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// generate a random array [x; m.0], x\in [0, 1)
    RandF32 {
        /// none
        i: (), 
        /// output symbol
        o: (&'t mut Symbol, ),
        /// length
        m: (usize, )
    },
    /// c(ai \* lbi\*lak\*lbk + bi \* lak\*lbk + ak \* lbk + bk) =
    ///     \sum_j a(ai \* laj \* lak + j \* lak + ak) \* b(bi \* lbj + j \* lbk + bk)
    MMulF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol), 
        /// (c, )
        o: (&'t mut Symbol, ),
        /// (lai, laj, lak, lbi, lbj, lbk), where laj == lbj
        m: (usize, usize, usize, usize, usize, usize, )
    },
    /// copy from one box to another
    Copy {
        /// source
        i: (&'t Symbol, ), 
        /// destination
        o: (&'t mut Symbol, ),
        /// none
        m: ()
    },
    /// transpose with given shape
    Transpose {
        /// source, (shape concatenated with permutation)
        i: (&'t Symbol, &'t Symbol, ),
        /// transposed symbol
        o: (&'t mut Symbol, ),
        /// (shape, permutation)
        m: ()
    },
    /// this faciliates _ => ... by making this enum non exhaustive 
    FallBack,
}

/* ------------------------------------------------------------------------------------------ */
/*                              traits for cooperative operations                             */
/* ------------------------------------------------------------------------------------------ */

/// data copy between peer devices
pub trait PeerCopy<D0: Device, D1: Device>
where Self: Device {
    /// copy data from d0 to d1, default implementation is [d0 -> host -> d1]
    fn pcpy(d0: &D0, d1: &D1, s0: &D0::Symbol, s1: &mut D1::Symbol)
        -> Result<(), (ComErr, Either<D0::DevErr, D1::DevErr>)> 
    {
        let data: Vec<u8> = match d0.dump(s0) {
            Err((ce, de)) => Err((ce, Either::A(de)))?,
            Ok(datbox) => datbox.into(),
        };
        match d1.load(data.into(), s1) {
            Err((ce, de)) => Err((ce, Either::B(de)))?,
            Ok(s1) => Ok(s1),
        }
    }
}