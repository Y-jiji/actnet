#![allow(warnings)]

use std::{ptr::null_mut, os::raw::c_int};

use crate::Void;
include!("../../cu-bind/cuda-driver.rs");

trait Wrap 
where Self: Sized {
    fn wrap<T>(self, v: T) -> Result<T, Self>;
}

impl Wrap for cudaError_enum {
    fn wrap<T>(self, v: T) -> Result<T, Self> {
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(v),
            e => Err(e)
        }
    }
}

/* -------------------------------------------------------------------------------------------------------------------------------- */

#[derive(Debug)]
struct DriverAPI;

impl DriverAPI {
    pub(crate)
    fn new() -> Result<DriverAPI, cudaError_enum> 
    {unsafe{cuInit(0)}.wrap(DriverAPI)}
}

#[cfg(test)]
mod check_driver_api {
    use super::DriverAPI;
    #[test]
    fn new() {
        let api_1 = DriverAPI::new().unwrap();
        let api_2 = DriverAPI::new().unwrap();
    }
}

/* -------------------------------------------------------------------------------------------------------------------------------- */

#[derive(Debug)]
pub(crate)
struct PhysDev {
    /// stack of (Context, Device)
    c: *mut CUctx_st,
    d: CUdevice
}

impl PhysDev {
    fn new(_api: &DriverAPI, nr: usize) -> Result<PhysDev, cudaError_enum> {
        let mut result = PhysDev { c: null_mut(), d: -1 };
        unsafe {
            cuDeviceGet(&mut result.d, nr as c_int).wrap(())?;
            cuCtxCreate_v2(&mut result.c, CUctx_flags_enum::CU_CTX_BLOCKING_SYNC as u32, result.d).wrap(())?;
        };
        Ok(result)
    }
}

impl Drop for PhysDev {
    fn drop(&mut self) {
        let r = unsafe{cuCtxDestroy_v2(self.c)}.wrap(());
        debug_assert!(r.is_ok(), "cuCtxDestroy failed");
    }
}

#[cfg(test)]
mod check_phys_dev {
    use super::*;

    #[test]
    fn new() {
        let api = DriverAPI::new().expect("Cannot initialize CUDA driver API");
        let phys_dev = PhysDev::new(&api, 0).unwrap();
        println!("{phys_dev:?}");
    }
}

/* -------------------------------------------------------------------------------------------------------------------------------- */

#[derive(Debug)]
pub(crate)
struct KerImgBuilder {
    p: *mut CUlinkState_st,
    o: Vec<CUjit_option_enum>,
    ov: Vec<*mut Void>,
}

#[derive(Debug)]
pub(crate)
struct KerImgData {
    /// pointer to data
    p: *mut Void,
    /// type
    t: CUjitInputType_enum,
    /// size
    s: usize,
    /// name
    n: String,
}

#[derive(Debug)]
pub(crate)
struct KerImgFile {
    /// file path
    p: String,
    /// type
    t: CUjitInputType_enum,
}

#[derive(Debug)]
pub(crate)
struct KerImg(*mut Void, usize);

impl KerImgBuilder {
    pub(crate)
    fn new() -> KerImgBuilder {
        KerImgBuilder{
            p: std::ptr::null_mut(), 
            o: Vec::new(), 
            ov: Vec::new(),
        }
    }
    pub(crate)
    fn option(mut self, o: CUjit_option_enum, ov: *mut Void) -> Result<KerImgBuilder, cudaError_enum> {
        debug_assert!(self.p.is_null(), "Option should be set before adding data!");
        if !self.p.is_null() {Err(cudaError_enum::CUDA_ERROR_ASSERT)}
        else {self.o.push(o); Ok(self)}
    }
    fn lazy_init(&mut self) -> Result<(), cudaError_enum> {
        let err = unsafe{cuLinkCreate_v2(
            self.o.len() as u32, 
            self.o.as_mut_ptr(), 
            self.ov.as_mut_ptr(), 
            &mut self.p)};
        return err.wrap(());
    }
    pub(crate)
    fn data(&mut self, _: &PhysDev, mut d: KerImgData) -> Result<&mut KerImgBuilder, cudaError_enum> {
        if self.p.is_null() { self.lazy_init()?; }
        let err = unsafe{cuLinkAddData_v2(
            self.p, 
            d.t, 
            d.p, 
            d.s, 
            d.n.as_str() as *const _ as *const i8, 
            0, null_mut(), null_mut()
        )};
        err.wrap(self)
    }
    pub(crate)
    fn file(&mut self, _: &PhysDev, mut f: KerImgFile) -> Result<&mut KerImgBuilder, cudaError_enum> {
        if self.p.is_null() { self.lazy_init()?; }
        let err = unsafe{cuLinkAddFile_v2(
            self.p, 
            f.t, 
            f.p.as_str() as *const _ as *const i8, 
            0, null_mut(), null_mut())};
        err.wrap(self)
    }
    pub(crate)
    fn build(mut self, _: &PhysDev) -> Result<KerImg, cudaError_enum> {
        if self.p.is_null() 
        { Err(cudaError_enum::CUDA_ERROR_INVALID_SOURCE)? }
        let mut kerimg = KerImg(null_mut(), 0usize);
        let err = unsafe{
            cuLinkComplete(self.p, &mut kerimg.0, &mut kerimg.1)};
        self.p = null_mut();
        return err.wrap(kerimg);
    }
}

impl Drop for KerImgBuilder {
    fn drop(&mut self) {
        #[cfg(test)] { println!("drop {self:?}"); return }
        debug_assert!(self.p.is_null(), "Initialized builder is dropped before building anything");
    }
}

#[cfg(test)]
mod check_kerimg_builder {
    use super::*;
    use std::{path::PathBuf, env::current_dir};

    #[test]
    fn init() {
        let api = DriverAPI::new().unwrap();
        let phys_dev = PhysDev::new(&api, 0);
        let mut builder = KerImgBuilder::new();
        builder.lazy_init().unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn data() {
        let api = DriverAPI::new().unwrap();
        let phys_dev = PhysDev::new(&api, 0).unwrap();
        let mut builder = KerImgBuilder::new();
        let p = include_str!("../../cu-target/test-case-1.ptx");
        builder.data(&phys_dev, 
            KerImgData {
                p: p as *const _ as *mut Void, 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX, 
                s: p.len(), n: String::new()
            }
        ).unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn file() {
        let api = DriverAPI::new().unwrap();
        let phys_dev = PhysDev::new(&api, 0).unwrap();
        let mut builder = KerImgBuilder::new();
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-1.ptx");
        builder.file(&phys_dev, 
            KerImgFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn link_and_build() {
        let api = DriverAPI::new().unwrap();
        let phys_dev = PhysDev::new(&api, 0).unwrap();
        let mut builder = KerImgBuilder::new();
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-1.ptx");
        builder.file(&phys_dev, 
            KerImgFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-2.ptx");
        builder.file(&phys_dev, 
            KerImgFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        println!("{builder:?}");
        let img = builder.build(&phys_dev);
        println!("{img:?}");
    }
}

/* -------------------------------------------------------------------------------------------------------------------------------- */

pub(crate)
struct CuModule {
    p: *mut CUmod_st,
}

impl CuModule {

}