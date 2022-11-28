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

#[derive(Debug)]
/// device function on device
/// i: input Symbol
/// o: output Symbol
/// m: meta data
pub enum Func<'t, Symbol: Debug + DTyped> {
    /// compute addition for each element
    /// c[i] = a[i] + b[i]
    AddF32 {
        /// input (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// output c (consume c)
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// compute subtraction for each element
    /// c[i] = a[i] - b[i]
    SubF32 {
        /// input (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// output c
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// compute multiplication for each element
    /// c[i] = a[i] * b[i]
    MulF32 {
        /// input (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// output c
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// compute division for each element
    /// c[i] = a[i] / b[i]
    DivF32 {
        /// input (a, b)
        i: (&'t Symbol, &'t Symbol, ), 
        /// output c
        o: (&'t mut Symbol, ),
        /// size of a, size of b
        m: (usize, )
    },
    /// generate a random array [x; m.0], x\in [0, 1)
    RandF32 {
        i: (), 
        o: (&'t mut Symbol, ),
        m: (usize, )
    },
    /// tensor contraction on a given dimension
    /// c[ai * lbi*lak*lbk + bi * lak*lbk + ak * lbk + bk] =
    ///     \sum_j a[ai * laj * lak + j * lak + ak] * b[bi * lbj + j * lbk + bk]
    MMulF32 {
        /// (a, b)
        i: (&'t Symbol, &'t Symbol), 
        /// (lai, laj, lak, lbi, lbj, lbk), where laj == lbj
        o: (&'t mut Symbol, ),
        m: (usize, usize, usize, usize, usize, usize, )
    },
    /// copy from one box to another, return this box and another
    Clone {
        i: (&'t Symbol, ), 
        o: (&'t mut Symbol, ),
        m: ()
    },
    /// this faciliates _ => ... by making this enum non exhaustive 
    FallBack,
}