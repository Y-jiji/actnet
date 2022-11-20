use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub enum DevFunc<DevBox: Debug> {
    /// compute addition for each element
    AddF32 {
        /// (a, b)
        read: (DevBox, DevBox), 
        /// c[i] = a[i] + b[i]
        write: DevBox, 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute subtraction for each element
    SubF32 {
        /// (a, b)
        read: (DevBox, DevBox), 
        /// c[i] = a[i] - b[i]
        write: DevBox, 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute multiplication for each element
    MulF32 {
        /// (a, b)
        read: (DevBox, DevBox), 
        /// c[i] = a[i] * b[i]
        write: DevBox, 
        /// size of a, size of b
        meta: (usize,)
    },
    /// compute division for each element
    DivF32 {
        /// (a, b)
        read: (DevBox, DevBox), 
        /// c[i] = a[i] / b[i]
        write: DevBox, 
        /// size of a, size of b
        meta: (usize,)
    },
    /// generate a random array [x; meta.0], x\in [0, 1)
    RandF32 {
        read: (), 
        write: DevBox, 
        meta: (usize,)
    },
    /// tensor contraction on a given dimension
    MMulF32 {
        /// (a, b)
        read: (DevBox, DevBox), 
        /// c[ai * lbi*lak*lbk + bi * lak*lbk + ak * lbk + bk] =
        ///     \sum_j a[ai * laj * lak + j * lak + ak] * b[bi * lbj + j * lbk + bk]
        write: DevBox, 
        /// (lai, laj, lak, lbi, lbj, lbk), where laj == lbj
        meta: (usize, usize, usize, usize, usize, usize)
    },
    /// copy from one box to another
    Cpy {
        read: DevBox, 
        write: DevBox, 
        meta: ()
    },
    FallBack,
}
