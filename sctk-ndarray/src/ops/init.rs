use crate::*;

trait Init<'a, D: Device, T: Num + Basic> 
where Self: Sized {
    type Err;
    /// initialize an ndarray with value
    fn with(value: T, shape: &[usize], device: &'a D) -> Result<Self, Self::Err>;
    /// initialize an ndarray with uniformly distributed random value in [0, upper)
    fn rand(upper: T, shape: &[usize], device: &'a D) -> Result<Self, Self::Err>;
}

impl<'a, D: Device, T: Num + Basic> Init<'static, D, T> for NDArray<'a, D, T> {
    type Err = TupErr<D>;
    fn rand(u: T, sh: &[usize], dv: &'a D) -> Result<Self, Self::Err> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty())?;
        match dv.emit(Func::Rand {i: (), o: (&mut sy, ), m: (u.wrap(), ln)}) {
            Err(e) => {dv.drop(sy)?; Err(e)},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv })
        }
    }
    fn with(v: T, sh: &[usize], dv: &'a D) -> Result<Self, Self::Err> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty())?;
        match dv.emit(Func::Rand {i: (), o: (&mut sy, ), m: (v.wrap(), ln)}) {
            Err(e) => {dv.drop(sy)?; Err(e)},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv })
        }
    }
}

#[cfg(test)]
mod check_init {
    #[test]
    fn rand() {
    }
}