use crate::*;

trait Init<'a, D: Device, T: Num + DevVal> 
where Self: Sized {
    type Err;
    /// initialize an ndarray with value
    fn fill(value: T, shape: &[usize], device: &'a D) -> Result<Self, Self::Err>;
    /// initialize an ndarray with uniformly distributed random value in [0, upper)
    fn rand(upper: T, shape: &[usize], device: &'a D) -> Result<Self, Self::Err>;
}

impl<'a, D: Device, T: Num + DevVal> Init<'static, D, T> for NDArray<'a, D, T> {
    type Err = DevErr<D>;
    fn rand(u: T, sh: &[usize], dv: &'a D) -> Result<Self, Self::Err> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty())?;
        match dv.emit(Func::RandUnif {i: (), o: (&mut sy, ), m: (ln, u.wrap())}) {
            Err(e) => {dv.drop(sy)?; Err(e)},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv, nil: false })
        }
    }
    fn fill(v: T, sh: &[usize], dv: &'a D) -> Result<Self, Self::Err> {
        let sh = sh.to_vec();
        let ln = sh.iter().product();
        let sz = T::msize(ln);
        let mut sy = dv.defn(sz, T::ty())?;
        match dv.emit(Func::Fill {i: (), o: (&mut sy, ), m: (ln, v.wrap())}) {
            Err(e) => {dv.drop(sy)?; Err(e)},
            Ok(()) => Ok(NDArray { sh, sy: nodrop(sy), ln, ty: phant(), dv, nil: false })
        }
    }
}

#[cfg(test)]
mod check_init {
    #[test]
    fn rand() {
    }
}