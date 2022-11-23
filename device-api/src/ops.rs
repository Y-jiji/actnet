use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32, 
    F64, 
    I32,
    I64, 
    Bool,
    FallBack
}

pub trait DTyped {
    fn dtype(&self) -> DType;
}

#[derive(Debug, Clone, Copy)]
pub enum Func<Symbol: Debug + DTyped> {
    /// compute addition for each element
    /// c[i] = a[i] + b[i]
    AddF32 {
        /// (a, b)
        read: (Symbol, Symbol), 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute subtraction for each element
    /// c[i] = a[i] - b[i]
    SubF32 {
        /// (a, b)
        read: (Symbol, Symbol), 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute multiplication for each element
    /// c[i] = a[i] * b[i]
    MulF32 {
        /// (a, b)
        read: (Symbol, Symbol), 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute division for each element
    /// c[i] = a[i] / b[i]
    DivF32 {
        /// (a, b)
        read: (Symbol, Symbol), 
        /// size of a, size of b
        meta: (usize,)
    },
    /// generate a random array [x; meta.0], x\in [0, 1)
    RandF32 {
        read: (), 
        meta: (usize,)
    },
    /// tensor contraction on a given dimension
    /// c[ai * lbi*lak*lbk + bi * lak*lbk + ak * lbk + bk] =
    ///     \sum_j a[ai * laj * lak + j * lak + ak] * b[bi * lbj + j * lbk + bk]
    MMulF32 {
        /// (a, b)
        read: (Symbol, Symbol), 
        /// (lai, laj, lak, lbi, lbj, lbk), where laj == lbj
        meta: (usize, usize, usize, usize, usize, usize)
    },
    /// copy from one box to another, return this box and another
    Clone {
        read: (Symbol, ), 
        meta: ()
    },
    /// this faciliates _ => ... by making this enum non exhaustive 
    FallBack,
}
