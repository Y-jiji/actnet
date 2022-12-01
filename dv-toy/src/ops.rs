use std::ptr::copy_nonoverlapping;

use crate::*;

pub(crate) fn add_f32(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) + *b.ptr::<f32>().add(i);
    }}
    Ok(())
}

pub(crate) fn sub_f32(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) - *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn mul_f32(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) * *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn div_f32(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, len: usize) -> Result<(), (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) / *b.ptr::<f32>().add(i);
    }}
    Ok(())
}


pub(crate) fn mmul_f32(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, meta: (usize, usize, usize, usize, usize, usize)) -> Result<(), (ComErr, ())> {
    let (lai, laj, lak, lbi, lbj, lbk) = meta;
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize * size_of::<f32>() != lai * laj * lak { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if b.msize * size_of::<f32>() != lbi * lbj * lbk { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if laj != lbj { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    assert!(c.msize == lai * lbi * lak * lbk * size_of::<f32>());
    let a_inner = a.ptr::<f32>() as *mut f32;
    let b_inner = b.ptr::<f32>() as *mut f32;
    let c_inner = c.ptr::<f32>() as *mut f32;
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

pub(crate) fn rand_f32(len: usize, a: &mut ToySymbol) -> Result<(), (ComErr, ())> {
    for i in 0..len {
        unsafe{*((a.ptr::<f32>() as *mut f32).add(i)) = rand::random::<f32>()}
    }
    Ok(())
}

pub(crate) fn copy(a: &ToySymbol, b: &mut ToySymbol) -> Result<(), (ComErr, ())> {
    unsafe{copy_nonoverlapping(a.inner, b.inner, a.msize)};
    Ok(())
}