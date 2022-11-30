use dvapi::*;
use std::alloc::*;

use std::fmt::Display;
use std::ptr::null_mut;

#[derive(Debug)]
pub struct DatBox {
    /// data type
    pub(crate) dtype: DType,
    /// inner pointer
    pub(crate) inner: *mut u8,
    /// memory size
    pub(crate) msize: usize,
}

impl Clone for DatBox {
    fn clone(&self) -> Self {
        let a = unsafe{Vec::from_raw_parts(self.inner, self.msize, self.msize)};
        let b = a.clone();
        let r = ByteConvert::from_byte(b, self.dtype);
        std::mem::forget(a); r
    }
}

impl GetDType for DatBox {fn dtype(&self) -> DType {self.dtype}}

impl ByteConvert for DatBox {
    fn as_byte(self) -> Vec<u8> {
        let r = unsafe{Vec::from_raw_parts(self.inner, self.msize, self.msize)};
        std::mem::forget(self); r
    }
    fn from_byte(mut x: Vec<u8>, ty: DType) -> Self {
        x.shrink_to_fit();
        let msize = x.len();
        DatBox { dtype: ty, inner: Vec::leak(x).as_ptr() as *mut u8, msize }
    }
}

impl VecConvert for DatBox {
    fn as_vec(self) -> WrapVec {        
        fn f<T>(_self: DatBox, _cons: fn(Vec<T>)-> WrapVec) -> WrapVec {
            let len = _self.msize/std::mem::size_of::<T>();
            let r = _cons(unsafe{Vec::from_raw_parts(_self.inner as *mut T, len, len)});
            std::mem::forget(_self); r
        }
        match self.dtype {
            DF32 => f(self, WF32),
            DF64 => f(self, WF64),
            DI32 => f(self, WI32),
            DI64 => f(self, WI64),
            DBool => f(self, WBool),
            _ => todo!()
        }
    }
    fn from_vec(x: WrapVec) -> DatBox {
        fn f<T>(mut x: Vec<T>, dtype: DType) -> DatBox {
            let x = {x.shrink_to_fit(); x};
            let msize = if dtype != DBool { x.len() * std::mem::size_of::<T>() } else { (x.len() + 7) / 8 };
            let inner = Vec::leak(x).as_ptr() as *mut u8;
            return DatBox { dtype, inner, msize }
        }
        match x {
            WF32(x) => f::<f32>(x, DF32),
            WF64(x) => f::<f64>(x, DF64),
            WI32(x) => f::<i32>(x, DI32),
            WI64(x) => f::<i64>(x, DI64),
            WBool(x) => f::<bool>(x, DBool),
            _ => todo!(),
        }
    }
}

impl Default for DatBox {
    fn default() -> Self {DatBox { dtype: DFallBack, inner: null_mut(), msize: 0 }}
}

impl Drop for DatBox {
    fn drop(&mut self) {
        drop(unsafe{Vec::from_raw_parts(self.inner, self.msize, self.msize)});
    }
}

/// format things like ndarray
fn print_rec<T: Display>(p: *const T, shape: &[usize], offset: usize, width: usize) -> String {
    if shape.len() == 0 {"[]".to_string()}
    else if shape.len() == 1 {unsafe {
        if shape[0] == 0 { return String::new() }
        let mut repr = String::new() + "[";
        for i in 0..usize::min(3, shape[0]-1) {
            let single_repr = format!("{} ", *p.add(i));
            repr += &vec![" "; width - single_repr.len()].concat();
            repr += &single_repr;
        }
        if shape[0] >= 5 { repr += " ... "; }
        let single_repr = format!("{}", *p.add(shape[0] - 1));
        repr += &vec![" "; width - single_repr.len()].concat();
        repr += &single_repr;
        repr + "]"
    }} else {
        let mut repr = String::new() + "[";
        let (l, shape) = shape.split_first().unwrap();
        let step = shape.iter().map(|a| *a).reduce(|a, b| a * b).unwrap();
        for i in 0..*l {unsafe {
            let p = p.add(i * step);
            if i != 0 { repr += &vec![" "; offset].concat(); }
            repr += &print_rec(p, shape, offset + 1, width);
            if i != *l-1 {repr += "\n"}
        }}
        repr + "]"
    }
}

// max width of numbers
fn width_rec<T: Display>(p: *const T, shape: &[usize]) -> usize {
    if shape.len() == 0 {0} 
    else if shape.len() == 1 {unsafe {
        if shape[0] == 0 { return 0 }
        let mut width = 0;
        for i in 0..usize::min(3, shape[0]-1) {
            width = usize::max(width, format!("{} ", *p.add(i)).len());
        }
        usize::max(width, format!("{} ", *p.add(shape[0] - 1)).len())
    }} else {
        let (l, shape) = shape.split_first().unwrap();
        let step = shape.iter().map(|a| *a).reduce(|a, b| a * b).unwrap();
        let mut width = 0;
        for i in 0..*l {unsafe {
            width = usize::max(width, width_rec(p.add(i*step), shape));
        }}
        width
    }
}

impl ArrayPrint for DatBox {
    fn print(&self, shape: Vec<usize>) -> String {
        let shape = shape.as_slice();
        let shape_prod = shape.iter().map(|x| *x).reduce(|a, b| a*b).unwrap_or(0);
        match self.dtype {
            DType::F32 => {
                let p = self.inner as *const f32;
                debug_assert!(shape_prod * 4 == self.msize);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::F64 => {
                let p = self.inner as *const f64;
                debug_assert!(shape_prod * 8 == self.msize);
                print_rec(p, shape, 1, width_rec(p, shape))
            }
            DType::I32 => {
                let p = self.inner as *const i32; 
                debug_assert!(shape_prod * 4 == self.msize);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::I64 => {
                let p = self.inner as *const i64; 
                debug_assert!(shape_prod * 8 == self.msize);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::Bool => {
                let p = self.inner as *const bool; 
                debug_assert!((shape_prod + 7) / 8 == self.msize);
                print_rec(p, shape, 1, width_rec(p, shape))    
            },
            _ => String::from("")
        }
    }
}

#[cfg(test)]
mod check_array_print {
    use super::*;
    use rand::*;

    #[test]
    fn print() {
        let a: Vec<_> = (0..(3*4*5)).map(|_| {random::<bool>()}).collect();
        let shape = [4,3,5];
        println!("{}", print_rec(a.as_ptr(), &shape, 1, width_rec(a.as_ptr(), &shape)));
        let a: Vec<_> = (0..(3*4*5)).map(|_| {random::<f32>()-0.5}).collect();
        let shape = [4,3,5];
        println!("{}", print_rec(a.as_ptr(), &shape, 1, width_rec(a.as_ptr(), &shape)));
        let a: Vec<_> = (0..(3*4*5)).map(|_| {random::<i64>()}).collect();
        let shape = [4,3,5];
        println!("{}", print_rec(a.as_ptr(), &shape, 1, width_rec(a.as_ptr(), &shape)));
    }

    #[test]
    fn new_and_drop() {
        let a: Vec<_> = (0..(3*4*5)).map(|_| {random::<bool>()}).collect();
        let a = DatBox::from_vec(WBool(a));
        println!("{a:?}");
        println!("{}", a.print(vec![4,3,5]));
    }
}