//! this module defines basic operations and operands for a device and correspondent data box

use std::fmt::Debug;
use crate::*;


/* ------------------------------------------------------------------------------------------ */
/*                                       operation types                                      */
/* ------------------------------------------------------------------------------------------ */

#[derive(Debug)]
/// device function on device, will get even larger in near future
/// (i: input Symbol;  o: output Symbol;  m: meta data)
pub enum Func<'t, S: Debug + Symbol> {
    /// c(i) <-- a(i) + b(i)
    Add {
        /// (a, b)
        i: (&'t S, &'t S, ), 
        /// (c, )
        o: (&'t mut S, ),
        /// size of a, size of b
        m: (usize, ),
    },
    /// c(i) <-- a(i) - b(i)
    Sub {
        /// (a, b)
        i: (&'t S, &'t S, ), 
        /// (c, )
        o: (&'t mut S, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// c(i) <-- a(i) * b(i)
    Mul {
        /// (a, b)
        i: (&'t S, &'t S, ), 
        /// (c, )
        o: (&'t mut S, ),
        /// length of a, length of b
        m: (usize, )
    },
    /// c(i) <-- a(i) / b(i)
    Div {
        /// (a, b)
        i: (&'t S, &'t S, ), 
        /// (c, )
        o: (&'t mut S, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// -a(i) <-- a(i)
    Neg {
        i: (&'t S, ),
        o: (&'t mut S, ),
        m: (usize, )
    },
    /// a random number in given range
    Rand {
        i: (),
        o: (&'t mut S, ),
        m: (WrapVal, usize, )
    },
    /// fill with a single scalar
    Fill {
        i: (),
        o: (&'t mut S, ),
        m: (WrapVal, usize, )
    },
    /// c(ai \* lbi\*lak\*lbk + bi \* lak\*lbk + ak \* lbk + bk) =
    ///     \sum_j a(ai \* laj \* lak + j \* lak + ak) \* b(bi \* lbj + j \* lbk + bk)
    MMul {
        /// (a, b)
        i: (&'t S, &'t S), 
        /// (c, )
        o: (&'t mut S, ),
        /// (lai, laj, lak, lbi, lbj, lbk), where laj == lbj
        m: (usize, usize, usize, usize, usize, usize, )
    },
    /// copy from one box to another
    Copy {
        /// source
        i: (&'t S, ), 
        /// destination
        o: (&'t mut S, ),
        /// none
        m: ()
    },
    /// transpose with given shape
    Transpose {
        /// source, (shape concatenated with permutation)
        i: (&'t S, &'t S, ),
        /// transposed symbol
        o: (&'t mut S, ),
        /// (shape, permutation)
        m: ()
    },
    /// this faciliates _ => ... by making this enum non exhaustive
    FallBack,
}