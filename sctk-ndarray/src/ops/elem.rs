//! elementwise operations

use crate::*;

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

use std::ops::{Add, Sub, Mul, Div};

macro_rules! impl_elem_op {
    ($op: tt, $OpName: tt, $op_name: tt) => {

        impl<'t, D: Device, T: DevVal> $OpName for &NDArray<'t, D, T> 
        where T: $OpName<Output=T> {
            type Output = Result<NDArray<'t, D, T>, DevErr<D>>;
            fn $op_name(self, rhs: Self) -> Self::Output {
                let (a, b) = if self.sh.len() > rhs.sh.len() {(self, rhs)} else {(rhs, self)};
                // shape and device check
                if a.dv != b.dv 
                { Err(FuncInvalidInputDifferentDevice(format!("{}: expected operands of {} on the same device, but found two: {:?} and {:?}", module_path!(), stringify!($op), a.dv, b.dv), D::DevErr::default()))? }
                if a.sh.split_at(a.len() - b.len()).1 != &b.sh 
                { Err(FuncInvalidInputShape(format!("{}: operands of {} cannot suffix broadcast: shorter shape vector {:?} is not a suffix of longer shape vector {:?}. ", module_path!(), stringify!($op), b.sh, a.sh), D::DevErr::default()))? }
                // calculate type and shape
                let (c_dv, c_ty, c_sh, c_ln) = (a.dv, a.ty, a.sh.clone(), a.ln);
                let mut c_sy = c_dv.defn(a.sy.msize(), a.sy.dtype())?;
                let (a, b) = (self, rhs);
                match c_dv.emit(Func::$OpName{
                    i: (&a.sy, &b.sy), 
                    o: (&mut c_sy, ), 
                    m: (a.len(), b.len())
                }) {
                    Err(e) => {c_dv.drop(c_sy)?; Err(e)},
                    Ok(()) => {Ok(NDArray { sy: nodrop(c_sy), ty: c_ty, dv: c_dv, ln: c_ln, sh: c_sh, nil: false })}
                }
            }
        }

    };
}

impl_elem_op!(+, Add, add);
impl_elem_op!(-, Sub, sub);
impl_elem_op!(*, Mul, mul);
impl_elem_op!(/, Div, div);

#[cfg(test)]
mod check_elem {
    use crate::{NDArray, dvtst::*};
    use crate::ops::init::*;

    #[test]
    fn add() {
        let dv = dvnew();
        let a = NDArray::fill(15.0, &[1, 3, 5], &dv).unwrap();
        println!("{a}");
    }
}