//! elementwise operations

use crate::*;
use std::ops::{Add, Sub, Mul, Div, Neg};

// @TODO: implement IPow for interger types, implement FPow for float types 

trait IPow
where Self: Sized {
    fn ipow(&self, p: i32) -> Self;
}

trait FPow
where Self: Sized {
    fn pow<Fp>(&self, p: Fp) -> Self;
    fn log<Fp>(&self, p: Fp) -> Self;
}

#[inline]
/// shape check
fn sh_check<D: Device>(a: &[usize], b: &[usize]) -> Result<Vec<usize>, TupErr<D>> {
    let mut ret = Vec::new();
    let b_suff = if a.len() < b.len() { ret=b.to_vec(); b.split_at(b.len() - a.len()).0 } else { b };
    let a_suff = if a.len() > b.len() { ret=a.to_vec(); a.split_at(a.len() - b.len()).0 } else { a };
    if a_suff != b_suff { Err((ComErr::FuncInvalidInputMeta, D::DevErr::default())) }
    else { Ok(ret) }
}

#[inline]
/// device check
fn dv_check<D: Device>(a: &D, b: &D) -> Result<(), TupErr<D>> {
    if a != b { Err((ComErr::FuncInvalidInputDifferentDevice, D::DevErr::default())) }
    else { Ok(()) }
}

macro_rules! impl_elem {
    ($UpperCase: tt, $LowerCase: tt) => {
/* ---------------------------------------------------------------------------------------- */
impl<'a, D: Device, T: Sized+'static> $UpperCase for &NDArray<'a, D, T> {
    type Output = Result<NDArray<'a, D, T>, TupErr<D>>;
    fn $LowerCase(self, rhs: Self) -> Self::Output {
        let lhs = self;
        dv_check(lhs.dv, rhs.dv)?;
        let sh = sh_check::<D>(&lhs.sh, &rhs.sh)?;
        let sz = usize::max(lhs.sy.msize(), rhs.sy.msize());
        let mut sy = lhs.dv.defn(sz, lhs.sy.dtype())?;
        match lhs.dv.emit(Func::$UpperCase { 
            i: (&lhs.sy, &rhs.sy, ), 
            o: (&mut sy, ), 
            m: (usize::min(lhs.len(), rhs.len()),) 
        }) {
            Err(e) => {
                // clear allocated symbol on error
                self.dv.drop(sy)?;
                Err(e)
            },
            Ok(()) => {
                // now ndarray is tracking symbol's lifetime
                let (ty, dv, ln) = (lhs.ty, lhs.dv, usize::max(lhs.len(), rhs.len()));
                Ok(NDArray { sy: nodrop(sy), ty, dv, sh, ln })
            }
        }
    }
}
/* ---------------------------------------------------------------------------------------- */
    };
}

impl_elem!(Div, div);
impl_elem!(Mul, mul);
impl_elem!(Sub, sub);
impl_elem!(Add, add);

impl<'a, D: Device, T: Sized+'static> Neg for &NDArray<'a, D, T> {
    type Output = Result<NDArray<'a, D, T>, TupErr<D>>;
    fn neg(self) -> Self::Output {
        let sh = self.sh.clone();
        let mut sy = self.dv.defn(self.sy.msize(), self.sy.dtype())?;
        match self.dv.emit(Func::Neg { 
            i: (&self.sy, ), 
            o: (&mut sy, ), 
            m: (self.ln, )
        }) {
            Err(e) => {
                self.dv.drop(sy)?;
                Err(e)
            },
            Ok(()) => {
                let (ty, dv, ln) = (self.ty, self.dv, self.ln);
                Ok(NDArray { sy: nodrop(sy), ty, dv, sh, ln })
            }
        }
    }
}

#[cfg(test)]
mod check_elem {
    use super::*;

    #[test]
    fn add() {
    }
}