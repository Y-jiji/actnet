/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;

/* ----------------------------------------------------------------------------------- */

use crate::Void;
use super::err::*;
use super::keeper::*;

/* ----------------------------------------------------------------------------------- */

use crate::raw::drv::{
    // cuda types
    CUjitInputType_enum,
    CUjit_option_enum,
    CUlinkState_st,
    // unsafe functions
    cuLinkCreate_v2,
    cuLinkAddFile_v2,
    cuLinkAddData_v2,
    cuLinkComplete,
    cuLinkDestroy,
};

/* ----------------------------------------------------------------------------------- */

// helpers to build kernel image
#[derive(Debug)]
pub struct JITInputData {
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
pub struct JITInputFile {
    /// file path
    p: String,
    /// type
    t: CUjitInputType_enum,
}

// kernel image with size, it should be dropped before cuda device uninitialize
#[derive(Debug)]
pub struct JITOutput<'a>(*mut Void, usize, &'a CudaDev);

/* ----------------------------------------------------------------------------------- */

#[derive(Debug)]
pub struct JITOutputBuilder<'a> {
    /// pointer to CUlinkState, where the JITCompiler will look into before invoking itself
    p: *mut CUlinkState_st,
    /// options of jit build
    o: Vec<CUjit_option_enum>,
    /// pointers to option values
    v: Vec<*mut Void>,
    /// immutable reference to a CudaDev, indicating this struct should die earlier than CudaDev
    d: &'a CudaDev,
}

impl<'a> JITOutputBuilder<'a> {
    /// create a new builder for JITOutput
    pub fn new(d: &'a CudaDev) -> JITOutputBuilder<'a> {
        JITOutputBuilder {
            p: std::ptr::null_mut(), 
            o: Vec::new(), 
            v: Vec::new(), d
        }
    }
    /// add an option to JIT Build (see CUDA documentation for details)
    pub fn option(mut self, o: CUjit_option_enum, v: *mut Void) -> Result<JITOutputBuilder<'a>, cudaError_enum> {
        debug_assert!(self.p.is_null(), "Option should be set before adding data!");
        if !self.p.is_null() {Err(cudaError_enum::CUDA_ERROR_ASSERT)}
        else {self.o.push(o); self.v.push(v); Ok(self)}
    }
    /// lazy init when data/file is added as source
    fn lazy_init(&mut self) -> Result<(), cudaError_enum> {
        let err = unsafe{cuLinkCreate_v2(
            self.o.len() as u32, 
            self.o.as_mut_ptr(), 
            self.v.as_mut_ptr(), 
            &mut self.p)};
        err.wrap(())
    }
    /// add data to source
    pub fn data(&mut self, d: JITInputData) -> Result<&mut JITOutputBuilder<'a>, cudaError_enum> {
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
    /// add file to source by file name
    pub fn file(&mut self, f: JITInputFile) -> Result<&mut JITOutputBuilder<'a>, cudaError_enum> {
        if self.p.is_null() { self.lazy_init()?; }
        let err = unsafe{cuLinkAddFile_v2(
            self.p, 
            f.t, 
            f.p.as_str() as *const _ as *const i8, 
            0, null_mut(), null_mut())};
        err.wrap(self)
    }
    /// build a kernel image on device
    pub fn build(mut self) -> Result<JITOutput<'a>, cudaError_enum> {
        if self.p.is_null() 
        { Err(cudaError_enum::CUDA_ERROR_INVALID_SOURCE)? }
        let mut kerimg = JITOutput(null_mut(), 0usize, self.d);
        let err = unsafe{
            cuLinkComplete(self.p, &mut kerimg.0, &mut kerimg.1);
            cuLinkDestroy(self.p)
        };
        self.p = null_mut();
        err.wrap(kerimg)
    }
}

impl<'a> Drop for JITOutputBuilder<'a> {
    fn drop(&mut self) {
        debug_assert!(self.p.is_null(), "Initialized builder is dropped before building anything");
    }
}

#[cfg(test)]
mod check_jit_output_builder {
    use super::*;
    use std::env::current_dir;

    #[test]
    fn init() {
        let zk = ZooKeeper::new().unwrap();
        let cd = CudaDev::new(&zk, 0).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        builder.lazy_init().unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn data() {
        let zk = ZooKeeper::new().unwrap();
        let cd = CudaDev::new(&zk, 0).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let p = include_str!("../../cu-target/test-case-1.ptx");
        builder.data(
            JITInputData {
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
        let zk = ZooKeeper::new().unwrap();
        let cd = CudaDev::new(&zk, 0).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-1.ptx");
        builder.file(
            JITInputFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn link_and_build() {
        let zk = ZooKeeper::new().unwrap();
        let cd = CudaDev::new(&zk, 0).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-1.ptx");
        builder.file( 
            JITInputFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-2.ptx");
        builder.file(
            JITInputFile {
                p: path.to_str().unwrap().to_owned(), 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX
            }
        ).unwrap();
        println!("{builder:?}");
        let img = builder.build();
        println!("{img:?}");
    }
}

/* ----------------------------------------------------------------------------------- */
