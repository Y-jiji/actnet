use std::ptr::copy_nonoverlapping;

mod elem;
mod gemm;
mod _rand;
mod copy;

pub(crate) use elem::*;
pub(crate) use _rand::*;
pub(crate) use gemm::*;
pub(crate) use copy::*;

