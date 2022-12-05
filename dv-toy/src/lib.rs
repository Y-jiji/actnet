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

    fn load(&self, datbox: Self::DatBox, symbol: &mut ToySymbol) -> Result<(), DevErr<Self>> {
        assert!(datbox.msize == symbol.msize);
        assert!(datbox.dtype == symbol.dtype);
        unsafe{copy_nonoverlapping(datbox.inner, symbol.inner, datbox.msize)};
        Ok(())
    }

    fn drop(&self, symbol: ToySymbol) -> Result<(), DevErr<Self>> {
        unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        std::mem::forget(symbol); Ok(())
    }

    fn defn(&self, size: usize, ty: DType) -> Result<ToySymbol, DevErr<Self>> {        
        let inner = Vec::leak({let mut x = vec![0u8; size]; x.shrink_to_fit(); x}).as_ptr() as *mut u8;
        Ok(ToySymbol { dtype: ty, inner, msize: size })
    }

    fn dump(&self, symbol: &ToySymbol) -> Result<Self::DatBox, DevErr<Self>> {
        let ty = symbol.dtype;
        let symvec = unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        let datbox = DatBox::from_byte(symvec.clone(), ty);
        std::mem::forget(symvec); Ok(datbox)
    }

    fn emit(&self, func: Func<ToySymbol>) -> Result<(), DevErr<Self>> {
        match func {
            Func::Add { i: (a, b), o: (c, ), m: (len_a, len_b) } => add(a, b, c, len_a, len_b),
            Func::Sub { i: (a, b), o: (c, ), m: (len_a, len_b) } => sub(a, b, c, len_a, len_b),
            Func::Mul { i: (a, b), o: (c, ), m: (len_a, len_b) } => mul(a, b, c, len_a, len_b),
            Func::Div { i: (a, b), o: (c, ), m: (len_a, len_b) } => div(a, b, c, len_a, len_b),
            Func::MMul { i: (a, b), o: (c, ), m } => mmul(a, b, c, m),
            Func::Copy { i: (a,), o: (b, ), m: () } => copy(a, b),
            Func::RandUnif { i: (), o: (a, ), m: (len, upper, ) } => rand_unif(upper, len, a),
            f => Err(DevErr::FuncNotimplemented(format!("{f:?}"), ()))
        }
    }
}

#[cfg(test)]
mod check_device_toy {
    use super::*;

    #[test]
    fn add_f32() {for _ in 0..100 {
    }}
}