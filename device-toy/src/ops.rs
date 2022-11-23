use std::ptr::copy_nonoverlapping;

use crate::*;

unsafe fn allocsiz(s: usize) -> *mut u8 {
    Vec::leak(vec![0u8; s]).as_ptr() as *mut u8
}

pub(crate) fn add_f32(a: Symbol, b: Symbol, len: usize) -> Result<Vec<Symbol>, (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    let c = Symbol {inner: unsafe{allocsiz(a.msize)}, msize: a.msize,dtype: DType::F32};
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) + *b.ptr::<f32>().add(i);
    }}
    Ok(vec![a, b, c])
}

pub(crate) fn sub_f32(a: Symbol, b: Symbol, len: usize) -> Result<Vec<Symbol>, (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    let c = Symbol {inner: unsafe{allocsiz(a.msize)}, msize: a.msize, dtype: DType::F32};
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) - *b.ptr::<f32>().add(i);
    }}
    Ok(vec![a, b, c])
}

pub(crate) fn mul_f32(a: Symbol, b: Symbol, len: usize) -> Result<Vec<Symbol>, (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    let c = Symbol {inner: unsafe{allocsiz(a.msize)}, msize: a.msize, dtype: DType::F32};
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) * *b.ptr::<f32>().add(i);
    }}
    Ok(vec![a, b, c])
}

pub(crate) fn div_f32(a: Symbol, b: Symbol, len: usize) -> Result<Vec<Symbol>, (ComErr, ())> {
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize != len * size_of::<f32>() { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    let c = Symbol {inner: unsafe{allocsiz(a.msize)}, msize: a.msize, dtype: DType::F32};
    for i in 0..len {unsafe{
        *c.ptr::<f32>().add(i) = *a.ptr::<f32>().add(i) / *b.ptr::<f32>().add(i);
    }}
    Ok(vec![a, b, c])
}

pub(crate) fn mmul_f32(a: Symbol, b: Symbol, meta: (usize, usize, usize, usize, usize, usize)) -> Result<Vec<Symbol>, (ComErr, ())> {
    let (lai, laj, lak, lbi, lbj, lbk) = meta;
    if a.msize != b.msize { Err((ComErr::FuncInvalidInputLength, ()))? }
    if a.msize * size_of::<f32>() != lai * laj * lak { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if b.msize * size_of::<f32>() != lbi * lbj * lbk { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if laj != lbj { Err((ComErr::FuncInvalidInputMeta, ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err((ComErr::FuncInvalidInputType, ()))? }
    let csize = lai * lbi * lak * lbk * size_of::<f32>();
    let c = Symbol { dtype: DType::F32, inner: unsafe{allocsiz(csize)}, msize: csize };
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
    Ok(vec![a, b, c])
}

pub(crate) fn rand_f32(len: usize) -> Result<Vec<Symbol>, (ComErr, ())> {
    let asize = len * size_of::<f32>();
    let a = Symbol { inner: unsafe{allocsiz(asize)}, msize: asize, dtype: DType::F32 };
    for i in 0..len {
        unsafe{*((a.ptr::<f32>() as *mut f32).add(i)) = rand::random::<f32>()}
    }
    Ok(vec![a])
}

pub(crate) fn clone(a: Symbol) -> Result<Vec<Symbol>, (ComErr, ())> {
    let b = Symbol { inner: unsafe{allocsiz(a.msize)}, msize: a.msize, dtype: DType::F32 };
    unsafe{copy_nonoverlapping(a.ptr::<f32>(), b.ptr::<f32>(), a.msize)};
    Ok(vec![a, b])
}