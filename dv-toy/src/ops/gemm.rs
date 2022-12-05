use crate::*;



pub(crate) fn mmul(a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, meta: (usize, usize, usize, usize, usize, usize, usize, usize)) -> Result<(), DevErr<Toy>> {
    let (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk) = meta;
    if a.msize != b.msize { Err(FuncInvalidInputLength(String::new(), ()))? }
    if a.msize * size_of::<f32>() != bat_a * lai * laj * lak { Err(FuncInvalidInputShape(String::new(), ()))? }
    if b.msize * size_of::<f32>() != bat_b * lbi * lbj * lbk { Err(FuncInvalidInputShape(String::new(), ()))? }
    if laj != lbj { Err(FuncInvalidInputShape(String::new(), ()))? }
    if a.dtype != DType::F32 || b.dtype != DType::F32 { Err(FuncInvalidInputType(String::new(), ()))? }
    assert!(c.msize == lai * lbi * lak * lbk * size_of::<f32>());
    for ai in 0..lai {
        for bi in 0..lbi {
            for ak in 0..lak {
                for bk in 0..lbk {
                    let mut tmp = 0f32;
                    for j in 0..lbj {
                        tmp += unsafe{*a.ptr::<f32>().add(ai * laj * lak + j * lak + ak) + *b.ptr::<f32>().add(bi * lbj + j * lbk + bk)};
                    }
                    unsafe{*c.ptr::<f32>().add(ai * lbi*lak*lbk + bi * lak*lbk + ak * lbk + bk) = tmp};
                }
            }
        }
    }
    Ok(())
}

