//! elementwise operations

use crate::*;
use std::ops::{Add, Sub, Mul, Div, Neg};

const MODULE_NAME: &str = "ops::elem";

// @TODO(Y-jiji: implement IPow for interger types, implement FPow for float types) 
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
fn sh_check<'t, D: Device, T: DevVal>
(a: &NDArray<'t, D, T>, b: &NDArray<'t, D, T>) -> Result<(Vec<usize>, usize), DevErr<D>> {
    let mut ret = (Vec::new(), 0usize);
    let b_suff = if a.len() < b.len() { ret=(b.sh.to_vec(), b.ln); b.sh.split_at(b.sh.len() - a.sh.len()).0 } else { b };
    let a_suff = if a.len() > b.len() { ret=(a.sh.to_vec(), a.ln); a.sh.split_at(a.sh.len() - b.sh.len()).0 } else { a };
    if a_suff != b_suff {
        let msg = format!(
            "element-wise add|sub|mul|div ... requires suffix-broadcasting. {}",
            "neither {a:?} nor {b:?} can be the other's suffix. [{SCTK_CRATE_NAME}::{MODULE_NAME}]"
        );
        Err(FuncInvalidInputMeta(msg, D::DevErr::default()))
    }
    else { Ok(ret) }
}

impl<'t, D: Device, T: DevVal> Add for &NDArray<'t, D, T> {
    type Output = Result<Self, DevErr<D>>;
    fn add(self, rhs: Self) -> Self::Output {
        let a = self;
        let b = rhs;
        let (c_sh, c_ln) = sh_check(a, b)?;
        todo!("@TODO(Y-jiji: implement element-wise add)");
    }
}


#[cfg(test)]
mod check_elem {
    use super::*;

    #[test]
    fn add() {
    }
}