use device_api::*;
use std::alloc::*;

use std::fmt::Display;
use std::mem::size_of;

#[derive(Debug)]
pub struct DatBox {
    /// data type
    pub(crate) dtype: DType,
    /// inner pointer
    pub(crate) inner: *mut u8,
    /// memory size
    pub(crate) msize: usize,
}

impl Drop for DatBox {
    fn drop(&mut self) {
        unsafe{dealloc(self.inner as *mut u8, Layout::from_size_align_unchecked(self.msize, 1))};
    }
}

impl From<Vec<f32>> for DatBox {
    fn from(v: Vec<f32>) -> Self {
        let v = Vec::leak(v);
        DatBox { dtype: DType::F32, inner: v as *mut _ as *mut u8, msize: v.len() * size_of::<f32>() }
    }
}

impl From<Vec<i32>> for DatBox {
    fn from(v: Vec<i32>) -> Self {
        let v = Vec::leak(v);
        DatBox { dtype: DType::I32, inner: v as *mut _ as *mut u8, msize: v.len() * size_of::<i32>() }
    }
}

impl From<Vec<f64>> for DatBox {
    fn from(v: Vec<f64>) -> Self {
        let v = Vec::leak(v);
        DatBox { dtype: DType::F64, inner: v as *mut _ as *mut u8, msize: v.len() * size_of::<f64>() }
    }
}

impl From<Vec<i64>> for DatBox {
    fn from(v: Vec<i64>) -> Self {
        let v = Vec::leak(v);
        DatBox { dtype: DType::I64, inner: v as *mut _ as *mut u8, msize: v.len() * size_of::<i64>() }
    }
}

#[cfg(test)]
impl Into<Vec<f32>> for DatBox {
    fn into(self) -> Vec<f32> {
        let r = unsafe{Vec::from_raw_parts(self.inner as *mut f32, self.msize / size_of::<f32>(), self.msize / size_of::<f32>())};
        std::mem::forget(self); r
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
                debug_assert!(shape_prod == self.msize * 8);
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
}