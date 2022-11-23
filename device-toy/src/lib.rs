use device_api::*;
use std::mem::size_of;

mod datbox;
use datbox::*;

mod symbol;
use symbol::*;

mod ops;
use ops::*;

#[derive(Debug)]
pub struct Toy;

impl<'a: 'b, 'b> Device<'a, 'b> for Toy {
    type Symbol = Symbol;
    type DatBox = DatBox;
    type DevErr = ();

    fn drop(&self, symbol: Self::Symbol) -> Result<(), (ComErr, Self::DevErr)> {
        unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        std::mem::forget(symbol); Ok(())
    }

    fn emit(&self, func: Func<Self::Symbol>) -> Result<Vec<Self::Symbol>, (ComErr, Self::DevErr)> {
        match func {
            Func::AddF32 { read: (a, b), meta: (len, ) } => add_f32(a, b, len), 
            Func::SubF32 { read: (a, b), meta: (len, ) } => sub_f32(a, b, len),
            Func::MulF32 { read: (a, b), meta: (len, ) } => mul_f32(a, b, len),
            Func::DivF32 { read: (a, b), meta: (len, ) } => div_f32(a, b, len),
            Func::RandF32 { read: (), meta: (len, ) } => rand_f32(len),
            Func::MMulF32 { read: (a, b), meta } => mmul_f32(a, b, meta),
            Func::Clone { read: (a, ), meta: () } => clone(a),
            _ => Err((ComErr::FuncNotimplemented, ()))
        }
    }

    fn dump(&self, symbol: Self::Symbol) -> Result<Self::DatBox, (ComErr, Self::DevErr)> {
        let r = Ok(DatBox { inner: symbol.inner, dtype: symbol.dtype, msize: symbol.msize });
        std::mem::forget(symbol); r
    }

    fn load(&self, datbox: Self::DatBox) -> Result<Self::Symbol, (ComErr, Self::DevErr)> {
        let r = Ok(Symbol { inner: datbox.inner, dtype: datbox.dtype,  msize: datbox.msize });
        std::mem::forget(datbox); r
    }
}

#[cfg(test)]
mod check_device_toy {
    use super::*;

    #[test]
    fn launch_add() {
        let toy = Toy;
        let a = toy.emit(Func::RandF32 { read: (), meta: (40_320, ) }).unwrap().into_iter().next().unwrap();
        let b = toy.emit(Func::RandF32 { read: (), meta: (40_320, ) }).unwrap().into_iter().next().unwrap();
        let mut abc = toy.emit(Func::AddF32 { read: (a, b), meta: (40_320, ) }).unwrap().into_iter();
        toy.drop(abc.next().unwrap()).unwrap();
        toy.drop(abc.next().unwrap()).unwrap();
        let c = abc.next().unwrap();
        let c = toy.dump(c).unwrap();
        println!("{}", c.print(vec![2,3,4,5,6,7,8]));
        let c: Vec<f32> = c.into();
        let c_mean: f32 = c.iter().sum::<f32>() / 40_320f32;
        // this is very very unlikely to fail, in the name of Markov's inequality
        assert!((c_mean - 1f32).abs() < 1e-2);
    }
}