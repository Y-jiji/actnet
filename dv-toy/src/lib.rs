use dvapi::*;
use std::{mem::size_of, ptr::copy_nonoverlapping};

mod datbox;
use datbox::*;

mod symbol;
use symbol::*;

mod ops;
use ops::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toy;

impl Device for Toy {
    type Symbol = ToySymbol;
    type DatBox = ToyDatBox;
    type DevErr = ();

    fn load(&self, datbox: Self::DatBox, symbol: &mut ToySymbol) -> Result<(), (ComErr, Self::DevErr)> {
        assert!(datbox.msize == symbol.msize);
        assert!(datbox.dtype == symbol.dtype);
        unsafe{copy_nonoverlapping(datbox.inner, symbol.inner, datbox.msize)};
        Ok(())
    }

    fn drop(&self, symbol: ToySymbol) -> Result<(), (ComErr, Self::DevErr)> {
        unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        std::mem::forget(symbol); Ok(())
    }

    fn defn(&self, size: usize, ty: DType) -> Result<ToySymbol, (ComErr, Self::DevErr)> {        
        let inner = Vec::leak({let mut x = vec![0u8; size]; x.shrink_to_fit(); x}).as_ptr() as *mut u8;
        Ok(ToySymbol { dtype: ty, inner, msize: size })
    }

    fn dump(&self, symbol: &ToySymbol) -> Result<Self::DatBox, (ComErr, Self::DevErr)> {
        let ty = symbol.dtype;
        let symvec = unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        let datbox = DatBox::from_byte(symvec.clone(), ty);
        std::mem::forget(symvec); Ok(datbox)
    }

    fn emit(&self, func: Func<ToySymbol>) -> Result<(), (ComErr, Self::DevErr)> {
        match func {
            Func::Add { i: (a, b), o: (c, ), m: (len, ) } => add_f32(a, b, c, len),
            Func::Sub { i: (a, b), o: (c, ), m: (len, ) } => sub_f32(a, b, c, len),
            Func::Mul { i: (a, b), o: (c, ), m: (len, ) } => mul_f32(a, b, c, len),
            Func::Div { i: (a, b), o: (c, ), m: (len, ) } => div_f32(a, b, c, len),
            Func::MMul { i: (a, b), o: (c, ), m } => mmul_f32(a, b, c, m),
            Func::Copy { i: (a,), o: (b, ), m: () } => copy(a, b),
            Func::Rand { i: (), o: (a, ), m: (len, ) } => rand_f32(len, a),
            _ => Err((ComErr::FuncNotimplemented, ()))
        }
    }
}

#[cfg(test)]
mod check_device_toy {
    use super::*;

    #[test]
    fn add_f32() {for _ in 0..100 {
        const CASE_SIZE: usize = 1<<15;
        let toy = Toy;
        let mut a = toy.defn(CASE_SIZE*4, DF32).unwrap();
        toy.emit(Func::RandF32 { i: (), o: (&mut a, ), m: (CASE_SIZE, ) }).unwrap();
        let mut b = toy.defn(CASE_SIZE*4, DF32).unwrap();
        toy.emit(Func::RandF32 { i: (), o: (&mut b, ), m: (CASE_SIZE, ) }).unwrap();
        let mut c = toy.defn(CASE_SIZE*4, DF32).unwrap();
        toy.emit(Func::AddF32 { i: (&a, &b), o: (&mut c, ), m: (CASE_SIZE, ) }).unwrap();
        let cvec = if let WF32(x) = toy.dump(&c).unwrap().as_vec() { x } else { todo!() };
        let mean: f32 = cvec.iter().sum::<f32>() / cvec.len() as f32;
        assert!((mean - 1.0).abs() < 0.01);
        toy.drop(a).unwrap();
        toy.drop(b).unwrap();
        toy.drop(c).unwrap();
    }}
}