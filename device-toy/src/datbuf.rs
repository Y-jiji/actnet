use device_api::*;
use std::alloc::*;

use std::fmt::Display;

#[derive(Debug)]
pub struct DatBuf {
    pub(crate) p: *mut (),
    pub(crate) s: usize,
    pub(crate) t: DType,
}

impl Drop for DatBuf {
    fn drop(&mut self) {
        unsafe{dealloc(self.p as *mut u8, Layout::from_size_align(self.s, 4).unwrap())};
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

impl ArrayPrint for DatBuf {
    fn print(&self, shape: Vec<usize>) -> String {
        let shape = shape.as_slice();
        let shape_prod = shape.iter().map(|x| *x).reduce(|a, b| a*b).unwrap_or(0);
        match self.t {
            DType::F32 => {
                let p = self.p as *const f32;
                debug_assert!(shape_prod * 4 == self.s);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::F64 => {
                let p = self.p as *const f64;
                debug_assert!(shape_prod * 8 == self.s);
                print_rec(p, shape, 1, width_rec(p, shape))
            }
            DType::I32 => {
                let p = self.p as *const i32; 
                debug_assert!(shape_prod * 4 == self.s);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::I64 => {
                let p = self.p as *const i64; 
                debug_assert!(shape_prod * 8 == self.s);
                print_rec(p, shape, 1, width_rec(p, shape))
            },
            DType::Bool => {
                let p = self.p as *const bool; 
                debug_assert!(shape_prod == self.s * 8);
                print_rec(p, shape, 1, width_rec(p, shape))    
            }
            _ => todo!()
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