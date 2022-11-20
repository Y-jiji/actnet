use device_api::*;
use std::{alloc::*, ops::Index};

pub struct Toy;

mod datbuf;
use datbuf::*;

#[derive(Debug)]
pub struct DevBox {
    p: *mut (),
    s: usize,
    t: DType,
}

impl DevBox {
    fn p_f32(&self) -> *mut f32 {
        self.p as *mut f32
    }
}

impl Device for Toy {
    type DatBuf = DatBuf;
    type DevBox = DevBox;
    type DevErr = ();

    fn delbox(&self, devbox: Self::DevBox) -> Result<(), (ComErr, Self::DevErr)> {
        unsafe{dealloc(devbox.p as *mut u8, Layout::from_size_align(devbox.s, 4).unwrap())};
        Ok(())
    }

    fn launch(&self, func: DevFunc<Self::DevBox>) -> Result<(), (ComErr, Self::DevErr)> {
        match func {
            DevFunc::AddF32 { read, write, meta } => {
                let l = meta.0;
                for i in 0..l {unsafe{
                    *write.p_f32().add(i) = 
                        *read.0.p_f32().add(i) + *read.1.p_f32().add(i);
                }}
                Ok(())
            },
            DevFunc::SubF32 { read, write, meta } => {
                let l = meta.0;
                for i in 0..l {unsafe{
                    *write.p_f32().add(i) = 
                        *read.0.p_f32().add(i) - *read.1.p_f32().add(i);
                }}
                Ok(())
            },
            DevFunc::MulF32 { read, write, meta } => {
                let l = meta.0;
                for i in 0..l {unsafe{
                    *write.p_f32().add(i) = 
                        *read.0.p_f32().add(i) * *read.1.p_f32().add(i);
                }}
                Ok(())
            },
            DevFunc::DivF32 { read, write, meta } => {
                let l = meta.0;
                for i in 0..l {unsafe{
                    *write.p_f32().add(i) = 
                        *read.0.p_f32().add(i) / *read.1.p_f32().add(i);
                }}
                Ok(())
            },
            _ => todo!("")
        }
    }

    fn newbox(&self, size: usize, dtype: DType) -> Result<Self::DevBox, (ComErr, Self::DevErr)> {
        let s = Layout::from_size_align(size, 4).unwrap();
        let p = unsafe{alloc(s)} as *mut ();
        let t = dtype;
        return Ok(DevBox { p, s: s.size(), t })
    }

    fn seebox(&self, devbox: Self::DevBox) -> Result<Self::DatBuf, (ComErr, Self::DevErr)> {
        let s = devbox.s;
        let p = unsafe{alloc(Layout::from_size_align(s, 4).unwrap())};
        let t = devbox.t;
        unsafe{std::ptr::copy_nonoverlapping(devbox.p as *mut u8, p, s)};
        Ok(DatBuf{ p: p as *mut (), s, t })
    }
}