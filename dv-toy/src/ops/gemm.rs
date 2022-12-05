use crate::*;
use std::ops::{Mul, AddAssign};
use std::iter::Product;
use itertools::Itertools;

fn mmul_branch<T: DevVal + AddAssign + Mul<Output = T>>(
    a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol,
    (bat_a, bat_b, li_a, lj_a, lk_a, li_b, lj_b, lk_b): (usize, usize, usize, usize, usize, usize, usize, usize)
) -> Result<(), DevErr<Toy>> {
    let len_tot_a = bat_a * li_a * lj_a * lk_a;
    if a.msize() != T::msize(len_tot_a)
    { Err(DevErr::FuncInvalidInputShape(format!("to read the value, real memory size {} and expected memory size {} should be equal", a.msize(), T::msize(len_tot_a)), ()))? }
    let len_tot_b = bat_b * li_b * lj_b * lk_b;
    if b.msize() != T::msize(len_tot_b)
    { Err(DevErr::FuncInvalidInputShape(format!("to read the value, real memory size {} and expected memory size {} should be equal", b.msize(), T::msize(len_tot_b)), ()))? }
    let len_tot_c = bat_a.max(bat_b) * li_a * lk_a * li_b * lk_b;
    if c.msize() != T::msize(len_tot_c)
    { Err(DevErr::FuncInvalidInputShape(format!("to write the value, real memory size {} and expected memory size {} should be equal", c.msize(), T::msize(len_tot_c)), ()))? }
    if bat_a % bat_b != 0 && bat_b % bat_a != 0
    { Err(DevErr::FuncInvalidInputShape(format!("to values to be suffix-broadcasting, batched operations should have one batch size multiple to another"), ()))? }
    if lj_a != lj_b
    { Err(DevErr::FuncInvalidInputShape(format!("for general matrix multiplication, the contracted dimension should have same length, but {lj_a} != {lj_b}"), ()))? }
    let a = |bt: usize, i: usize, j: usize, k: usize| unsafe{a.ptr::<T>().add((bt % bat_a) * li_a * lj_a * lk_a + i * lj_a * lk_a + j * lk_a + k)};
    let b = |bt: usize, i: usize, j: usize, k: usize| unsafe{b.ptr::<T>().add((bt % bat_b) * li_b * lj_b * lk_b + i * lj_b * lk_b + j * lk_b + k)};
    let c = |bt: usize, i_a: usize, i_b: usize, k_a: usize, k_b: usize|
        unsafe{c.ptr::<T>().add((bt % bat_a.max(bat_b)) * li_a * li_b * lk_a * lk_b + i_a * li_b * lk_a * lk_b + i_b * lk_a * lk_b + k_a * lk_b + k_b)};
    for bt in 0..bat_a.max(bat_b) {
    for i_a in 0..li_a {
    for k_a in 0..lk_a {
    for i_b in 0..li_b {
    for k_b in 0..lk_b {
    for j in 0..lj_a {
        unsafe{*c(bt, i_a, i_b, k_a, k_b) += *a(bt, i_a, j, k_a) * *b(bt, i_b, j, k_b)};
    }}}}}}
    Ok(())
}

pub(crate) fn mmul(
    a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, 
    (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk): (usize, usize, usize, usize, usize, usize, usize, usize)
) -> Result<(), DevErr<Toy>> {
    if a.dtype() != c.dtype()
    { Err(DevErr::FuncInvalidInputType(format!("input and output should have the same type, but we found {:?} and {:?}", a.dtype(), c.dtype()), ()))? }
    if b.dtype() != c.dtype() 
    { Err(DevErr::FuncInvalidInputType(format!("input and output should have the same type, but we found {:?} and {:?}", b.dtype(), c.dtype()), ()))? }
    match c.dtype() {
        DF32 => mmul_branch::<f32>(a, b, c, (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk)),
        DI32 => mmul_branch::<i32>(a, b, c, (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk)),
        DF64 => mmul_branch::<f64>(a, b, c, (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk)),
        DI64 => mmul_branch::<i64>(a, b, c, (bat_a, bat_b, lai, laj, lak, lbi, lbj, lbk)),
        e => Err(DevErr::FuncInvalidInputType(format!("general matrix multiplication only works on data types with addition and multiplication, but we found {e:?}"), ()))
    }
}

fn einsum_branch<T: DevVal + AddAssign + Mul<Output = T> + Product<T> + Copy>(
    x: &[&ToySymbol], y: &mut ToySymbol,
    (bat, sh, idx): (&[usize], &[&[usize]], &[&[(usize, usize)]])
) -> Result<(), DevErr<Toy>> {
    let l = x.len();
    let x = &[&*x, &[&*y]].concat();
    // of[k]: offset multiplier for each dimension for x[k], x[x.len()] represents y
    let of: Vec<_> = (0..l).map(|k| {
        let sh = sh[k];
        let mut of = vec![1usize; sh.len()+1];
        for j in 1..(sh.len()+1) { of[sh.len()-j-1] = of[sh.len()-j]*sh[sh.len()-j] }
        return of
    }).collect();
    let x = |k: usize, bt: usize, i: &[usize]| -> *mut T {
        let (x, of) = (x[k], &of[k]);
        let i = i.iter().enumerate().map(
            |(j, i)| idx[j].iter()
            .filter_map(|&(_k, d)| if _k != k { None } else { Some(of[d] * i) }).sum::<usize>()
        ).sum::<usize>();
        unsafe{x.ptr::<T>().add((bt % bat[k])*of[0]+i)}
    };
    let mut shape_ok = true;
    let glbl_sh: Vec<_> = idx.iter().map_while(|x| {
        let ok = x.iter().all(|(k, d)| sh[*k][*d] == sh[x[0].0][x[0].1]);
        if !ok { shape_ok = false; None }
        else { let (k, d) = x[0]; Some(sh[k][d]) }
    }).collect();
    if !shape_ok { Err(DevErr::FuncInvalidInputShape(format!("length of paired dimensions doesn't match, please check your einsum index"), ()))? }
    for bt in 0..*bat.iter().max().unwrap() {
        let i_space = glbl_sh.iter().map(|x| 0..*x).multi_cartesian_product();
        for i in i_space {
            unsafe{*x(l, bt, &i) += (0..l).map(|j| {*x(j, bt, &i)}).product()}
        }
    }
    Ok(())
}

pub(crate) fn einsum(
    x: &[&ToySymbol], 
    y: &mut ToySymbol,
    (bat, sh, idx): (&[usize], &[&[usize]], &[&[(usize, usize)]])
) -> Result<(), DevErr<Toy>> {
    match y.dtype() {
        DF32 => einsum_branch::<f32>(x, y, (bat, sh, idx)),
        DF64 => einsum_branch::<f64>(x, y, (bat, sh, idx)),
        DI32 => einsum_branch::<i32>(x, y, (bat, sh, idx)),
        DI64 => einsum_branch::<i64>(x, y, (bat, sh, idx)),
        e => Err(DevErr::FuncInvalidInputType(format!("general matrix multiplication only works on data types with addition and multiplication, but we found {e:?}"), ()))?
    }
}