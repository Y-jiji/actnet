use std::fmt::Debug;
use std::result::*;

#[derive(Debug)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

/// common error format
#[derive(Debug)]
pub enum ComErr {
    /// (wanted, total)
    OutOfMemory(usize, usize, ),
}

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

pub trait ArrayPrint {
    fn print(&self, shape: Vec<usize>) -> String {todo!("ArrayPrint::print({shape:?})");}
}

pub trait Device
where Self::DevBox: Debug,
      Self::DatBuf: ArrayPrint {

    /// device box in device
    type DevBox;

    /// data buffer on host
    type DatBuf;

    /// device error
    type DevErr;

    /// launch a device function
    fn launch(&self, func: DevFunc<Self::DevBox>) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("launch({func:?})") }

    /// allocate a new box on this device
    fn newbox(&self, size: usize, dtype: DType) -> Result<Self::DevBox, (ComErr, Self::DevErr)>
    { todo!("newbox({size:?})") }

    /// delete a box
    fn delbox(&self, devbox: Self::DevBox) -> Result<(), (ComErr, Self::DevErr)>
    { todo!("delbox({devbox:?})") }

    /// inspect devbox (dump bytes into a data buffer)
    fn seebox(&self, devbox: Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)>
    { todo!("seebox({devbox:?})") }
}