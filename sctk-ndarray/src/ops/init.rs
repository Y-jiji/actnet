use crate::*;

pub trait Init<'a, D: Device, T: DevVal> 
where Self: Sized {
    /// initialize an ndarray with value
    fn fill(value: T, shape: &[usize], device: &'a D) -> Result<Self, DevErr<D>>;
    /// initialize an ndarray with uniformly distributed random value in [0, upper)
    fn unif(upper: T, shape: &[usize], device: &'a D) -> Result<Self, DevErr<D>>;
}

impl<'a, D: Device, T: DevVal> Init<'a, D, T> for NDArray<'a, D, T> {
    fn unif(u: T, sh: &[usize], dv: &'a D) -> Result<Self, DevErr<D>> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty()).map_err(|e| e.prefix(format!("{}: ", module_path!())))?;
        match dv.emit(Func::RandUnif {i: (), o: (&mut sy, ), m: (ln, u.wrap())}) {
            Err(e) => {dv.drop(sy)?; Err(e.prefix(format!("{}: ", module_path!())))},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv, nil: false })
        }
    }
    fn fill(v: T, sh: &[usize], dv: &'a D) -> Result<Self, DevErr<D>> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty()).map_err(|e| e.prefix(format!("{}: ", module_path!())))?;
        match dv.emit(Func::Fill {i: (), o: (&mut sy, ), m: (ln, v.wrap())}) {
            Err(e) => {dv.drop(sy)?; Err(e.prefix(format!("{}: ", module_path!())))},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv, nil: false })
        }
    }
}

#[cfg(test)]
mod check_init {
    use crate::{dvtst::*, NDArray};
    use super::*;

    #[test]
    fn unif() {
        let dv = dvnew();
        let a = NDArray::unif(true, &[1, 3, 5], &dv).unwrap();
        println!("{a}");
    }

    #[test]
    fn fill() {
        let dv = dvnew();
        let a = NDArray::fill(true, &[1, 3, 5], &dv).unwrap();
        println!("{a}");
    }
}