use std::ptr::copy_nonoverlapping;

use crate::*;

pub(crate) fn add(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), DevErr<Toy>> {
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize != len * size_of::<f32>() { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) + *b.ptr::<f32>().add(i);
    }}
    Ok(())
}

pub(crate) fn sub(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), DevErr<Toy>> {
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize != len * size_of::<f32>() { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) - *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn mul(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), DevErr<Toy>> {
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize != len * size_of::<f32>() { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) * *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn div(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), DevErr<Toy>> {
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize != len * size_of::<f32>() { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) / *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn mmul(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, meta: (usize, usize, usize, usize, usize, usize, usize)) -> Result<(), DevErr<Toy>> {
    let (bat, lai, laj, lak, lbi, lbj, lbk) = meta;
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize * size_of::<f32>() != bat * lai * laj * lak { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if b.msize * size_of::<f32>() != bat * lbi * lbj * lbk { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if laj != lbj { Err(FuncInvalidInputMeta(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    assert!(c.msize == lai * lbi * lak * lbk * size_of::<f32>());
    let a_inner = a.ptr::<f32>();
    let b_inner = b.ptr::<f32>();
    let c_inner = c.ptr::<f32>();
    for ai in 0..lai {
        for bi in 0..lbi {
            for ak in 0..lak {
                for bk in 0..lbk {
                    let mut tmp = 0f32;
                    for j in 0..lbj {
                        tmp += unsafe{*a_inner.add(ai * laj * lak + j * lak + ak) + *b_inner.add(bi * lbj + j * lbk + bk)};
                    }
                    unsafe{*c_inner.add(ai * lbi*lak*lbk + bi * lak*lbk + ak * lbk + bk) = tmp};
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn rand_f<T>(upper: T, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> 
where rand::distributions::Standard: rand::distributions::Distribution<T>, 
      T: std::ops::Mul<Output=T> + Copy {
    for i in 0..len {
        unsafe{*a.ptr::<T>().add(i) = upper * rand::random::<T>()}
    }
    Ok(())
}

pub(crate) fn rand_i<T>(upper: T, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> 
where rand::distributions::Standard: rand::distributions::Distribution<T>,
      T: std::ops::Rem<Output=T> + Copy {
    for i in 0..len {
        unsafe{*a.ptr::<T>().add(i) = rand::random::<T>() % upper}
    }
    Ok(())
}

pub(crate) fn rand_b(upper: bool, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    if upper {unsafe{
        let len = bool::msize(len);
        for i in 0..(len/16) {*a.ptr::<u128>().add(i) = rand::random::<u128>()}
        for i in 0..(len%16) {*a.ptr::<u8>().add(i) = rand::random::<u8>()}
    }} else {unsafe{
        std::slice::
        from_raw_parts_mut(a.ptr::<u8>(), bool::msize(len)).fill(0);
    }}
    Ok(())
}

pub(crate) fn copy(a: &ToySymbol, b: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    unsafe{copy_nonoverlapping(a.inner, b.inner, a.msize)};
    Ok(())
}
