use dvapi::WrapVal;
use crate::*;

fn fill_branch<T: Copy+DevVal>(value: WrapVal, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    let value = T::unwrap(value);
    if a.msize() != T::msize(len) { Err(DevErr::FuncInvalidInputShape(format!("dv-toy::{} -> memory size and expected length doesn't match", module_path!()), ()))? }
    let mut a = unsafe{Vec::from_raw_parts(a.ptr::<T>(), len, len)};
    for i in 0..len { a[i] = value; }
    std::mem::forget(a);
    Ok(())
}

pub(crate) fn fill(value: WrapVal, len: usize, a: &mut ToySymbol)
-> Result<(), DevErr<Toy>> {
    if a.dtype() != value.dtype() { Err(DevErr::FuncInvalidInputType(format!("value and operated symbol should have the same type, but found {:?} and {:?}", a.dtype(), value.dtype()), ()))? }
    match a.dtype() {
        DBool => {
            let value = bool::unwrap(value);
            if a.msize() != bool::msize(len) { Err(DevErr::FuncInvalidInputShape(format!("dv-toy::{} -> memory size and expected length doesn't match", module_path!()), ()))? }
            let len = (len+7) >> 8;
            let mut a = unsafe {Vec::from_raw_parts(a.ptr::<u8>(), len, len)};
            let value = if value { !0u8 } else { 0u8 };
            for i in 0..len { a[i] = value; }
            std::mem::forget(a);
            Ok(())
        },
        DF32 => fill_branch::<f32>(value, len, a),
        DF64 => fill_branch::<f64>(value, len, a),
        DI32 => fill_branch::<i32>(value, len, a),
        DI64 => fill_branch::<i64>(value, len, a),
        e => Err(DevErr::FuncInvalidInputType(format!("Func::Fill is not implemented for {e:?}"), ()))
    }
}