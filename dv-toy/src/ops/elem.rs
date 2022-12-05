use crate::*;
use std::ops::{Add, Sub, Mul, Div};

macro_rules! impl_elem_op {
    ($f_name: tt, $f_branch_name: tt, $OpTrait: tt, $op: tt) => {

        fn $f_branch_name<T: DevVal + $OpTrait<Output=T>>(
            a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, 
            bat_a: usize, bat_b: usize
        ) -> Result<(), DevErr<Toy>> {
            if a.msize() != T::msize(bat_a) 
            { Err(DevErr::FuncInvalidInputShape(format!("to read the value, real memory size {} and expected memory size {} should be equal", a.msize(), T::msize(bat_a)), ()))? }
            if b.msize() != T::msize(bat_b) 
            { Err(DevErr::FuncInvalidInputShape(format!("to read the value, real memory size {} and expected memory size {} should be equal", b.msize(), T::msize(bat_b)), ()))? }
            let len_c = usize::max(bat_a, bat_b);
            if c.msize() != T::msize(len_c) 
            { Err(DevErr::FuncInvalidInputShape(format!("to write the value, real memory size {} and expected memory size {} should be equal", c.msize(), T::msize(len_c)), ()))? }
            if bat_a % bat_b != 0 && bat_b % bat_a != 0
            { Err(DevErr::FuncInvalidInputShape(format!("for input values to broadcast, shape of one value should be a suffix of another value"), ()))? }
            for i in 0..len_c {unsafe{
                *c.ptr::<T>().add(i) = *a.ptr::<T>().add(i % bat_a) $op *b.ptr::<T>().add(i % bat_b);
            }}
            Ok(())
        }

        pub(crate) fn $f_name(
            a: &ToySymbol, b: &ToySymbol, c: &mut ToySymbol, 
            bat_a: usize, bat_b: usize
        ) -> Result<(), DevErr<Toy>> {
            if c.dtype() != a.dtype() {return Err(DevErr::FuncInvalidInputType(format!("input and output should have the same type, but there are {:?} {:?}", a.dtype(), c.dtype()), ()))}
            if c.dtype() != b.dtype() {return Err(DevErr::FuncInvalidInputType(format!("input and output should have the same type, but there are {:?} {:?}", b.dtype(), c.dtype()), ()))}
            match c.dtype() {
                DF32 => $f_branch_name::<f32>(a, b, c, bat_a, bat_b),
                DF64 => $f_branch_name::<f64>(a, b, c, bat_a, bat_b),
                DI32 => $f_branch_name::<i32>(a, b, c, bat_a, bat_b),
                DI64 => $f_branch_name::<i64>(a, b, c, bat_a, bat_b),
                e => Err(DevErr::FuncInvalidInputType(format!("input and output should have {}able types, found {e:?}", stringify!($f_name)), ())),
            }
        }

    };
}

impl_elem_op!(add, add_branch, Add, +);
impl_elem_op!(sub, sub_branch, Sub, -);
impl_elem_op!(mul, mul_branch, Mul, *);
impl_elem_op!(div, div_branch, Div, /);