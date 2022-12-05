use dvapi::*;
use std::ptr::copy_nonoverlapping;

mod datbox;
use datbox::*;

mod symbol;
use symbol::*;

mod ops;
use ops::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Toy;

impl Toy { pub fn new() -> Toy { Toy } }

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

    fn defn(&self, msize: usize, ty: DType) -> Result<ToySymbol, DevErr<Self>> {        
        let inner = Vec::leak({let mut x = vec![0u8; msize]; x.shrink_to_fit(); x}).as_ptr() as *mut u8;
        Ok(ToySymbol { dtype: ty, inner, msize })
    }

    fn dump(&self, symbol: &ToySymbol) -> Result<Self::DatBox, DevErr<Self>> {
        let ty = symbol.dtype;
        let symvec = unsafe{Vec::from_raw_parts(symbol.inner, symbol.msize, symbol.msize)};
        let datbox = DatBox::from_byte(symvec.clone(), ty);
        std::mem::forget(symvec); Ok(datbox)
    }

    fn emit(&self, func: Func<ToySymbol>) -> Result<(), DevErr<Self>> {
        // emit dispatches function calls to implementations
        match func {
            Func::Add { i: (a, b), o: (c, ), m: (bat_a, bat_b) } => add(a, b, c, bat_a, bat_b),
            Func::Sub { i: (a, b), o: (c, ), m: (bat_a, bat_b) } => sub(a, b, c, bat_a, bat_b),
            Func::Mul { i: (a, b), o: (c, ), m: (bat_a, bat_b) } => mul(a, b, c, bat_a, bat_b),
            Func::Div { i: (a, b), o: (c, ), m: (bat_a, bat_b) } => div(a, b, c, bat_a, bat_b),
            Func::MMul { i: (a, b), o: (c, ), m } => mmul(a, b, c, m),
            Func::EinSum { i: x, o: (y,), m: (bat, sh, idx) } => einsum(x, y, (bat, sh, idx)),
            Func::Copy { i: (a,), o: (b, ), m: () } => copy(a, b),
            Func::RandUnif { i: (), o: (a, ), m: (len, upper, ) } => rand_unif(upper, len, a),
            Func::Fill { i: (), o: (a, ), m: (len, value, ) } => fill(value, len, a),
            f => Err(DevErr::FuncNotimplemented(format!("{f:?}"), ()))
        }
    }

    fn name(&self) -> String {
        format!("Toy(thread_id: {:?})", std::thread::current().id())
    }
}

#[cfg(test)]
mod check_device_toy {
    use super::*;

    #[test]
    fn add_f32() {
        let toy = Toy::new();
        let mut a = toy.defn(f32::msize(256), f32::ty()).unwrap();
        let mut b = toy.defn(f32::msize(3*256), f32::ty()).unwrap();
        let mut c = toy.defn(f32::msize(3*256), f32::ty()).unwrap();
        toy.load(<Toy as Device>::DatBox::from_vec(WrapVec::F32(vec![1.0; 256])), &mut a).unwrap();
        toy.load(<Toy as Device>::DatBox::from_vec(WrapVec::F32(vec![2.0; 3*256])), &mut b).unwrap();
        toy.load(<Toy as Device>::DatBox::from_vec(WrapVec::F32(vec![0.0; 3*256])), &mut c).unwrap();
        toy.emit(Func::Add { i: (&a, &b), o: (&mut c, ), m: (256, 3*256) }).unwrap();
        let c = {
            let dat = toy.dump(&c);
            toy.drop(c).unwrap();
            match dat.unwrap().as_vec() { WrapVec::F32(c)=>c, _ => panic!("unexpected type") }
        };
        assert!(c == vec![3.0; 3*256]);
        toy.drop(a).unwrap();
        toy.drop(b).unwrap();
    }
}