/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;

/* ----------------------------------------------------------------------------------- */

use crate::Void;
use super::err::*;
use super::zk::*;

/* ----------------------------------------------------------------------------------- */

pub(crate)
use crate::raw::drv::{
    // cuda types
    CUjitInputType_enum,
    CUjit_option_enum,
    CUlinkState_st,
    CUmod_st,
    CUfunc_st,
    // unsafe functions
    cuLinkCreate_v2,
    cuLinkAddFile_v2,
    cuLinkAddData_v2,
    cuLinkComplete,
    cuLinkDestroy,
    cuModuleLoadData,
    cuModuleGetFunction,
};

/* ----------------------------------------------------------------------------------- */

// helpers to build kernel image
#[derive(Debug)]
pub struct JITInputData {
    /// pointer to data
    pub p: *mut Void,
    /// type
    pub t: CUjitInputType_enum,
    /// size
    pub s: usize,
    /// name
    pub n: String,
}

#[derive(Debug)]
pub struct JITInputFile {
    /// file path
    pub p: String,
    /// type
    pub t: CUjitInputType_enum,
}

// kernel image with size, it should be dropped before cuda device uninitialize
#[derive(Debug)]
pub struct JITOutput<'a>(*mut Void, usize, &'a CuDev);

/* ----------------------------------------------------------------------------------- */

#[derive(Debug)]
pub struct JITOutputBuilder<'a> {
    /// pointer to CUlinkState, where the JITCompiler will look into before invoking itself
    pub(crate)
    p: *mut CUlinkState_st,
    /// options of jit build
    o: Vec<CUjit_option_enum>,
    /// pointers to option values (this is error-prone, be careful)
    v: Vec<*mut Void>,
    /// immutable reference to a CudaDev, indicating this struct should be dropped earlier than corresponding CudaDev
    cd: &'a CuDev,
}

impl<'a> JITOutputBuilder<'a> {
    /// create a new builder for JITOutput
    pub fn new(cd: &'a CuDev) -> JITOutputBuilder<'a> {
        JITOutputBuilder {
            p: std::ptr::null_mut(), 
            o: Vec::new(), 
            v: Vec::new(), cd
        }
    }
    /// add an option to JIT Build (see CUDA documentation for details)
    pub fn option(&mut self, o: CUjit_option_enum, v: *mut Void) -> Result<&mut JITOutputBuilder<'a>, cudaError_enum> {
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
        let mut kerimg = JITOutput(null_mut(), 0usize, self.cd);
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
        if !self.p.is_null() {
            unsafe{ cuLinkDestroy(self.p); }
        }
    }
}

#[cfg(test)]
mod check_jit_output_builder {
    use super::*;
    use std::env::current_dir;

    #[test]
    fn init() {
        let zk = ZooKeeper::new().unwrap();
        let cd = CuDev::new(&zk, 0, 1024).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        builder.lazy_init().unwrap();
        println!("{builder:?}");
        builder.p = null_mut();
    }

    #[test]
    fn data() {
        let zk = ZooKeeper::new().unwrap();
        let cd = CuDev::new(&zk, 0, 1024).unwrap();
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
        let cd = CuDev::new(&zk, 0, 1024).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let path = current_dir().unwrap();
        let path = path.join("cu-target").join("test-case-1.ptx");
        assert!(path.exists());
        println!("{path:?}");
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
        let cd = CuDev::new(&zk, 0, 1024).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let mut add_f = |target: &str| {
            let path = current_dir().unwrap();
            let path = path.join("cu-target").join(target);
            assert!(path.exists());
            println!("{path:?}");
            builder.file(
                JITInputFile {
                    p: path.to_str().unwrap().to_owned(), 
                    t: CUjitInputType_enum::CU_JIT_INPUT_PTX
                }
            ).unwrap();
        };
        add_f("test-case-1.ptx");
        add_f("test-case-2.ptx");
        println!("{builder:?}");
        let img = builder.build().unwrap();
        println!("{img:?}");
    }
}

/* ----------------------------------------------------------------------------------- */

/// a jit caller (to acquire function handle)
#[derive(Debug)]
pub struct JITCaller<'a> {
    p: *mut CUmod_st,
    cd: &'a CuDev,
    s: usize,
}

/// a function handle with restricted lifetime
#[derive(Debug, Clone)]
pub struct FuncHandle<'a> {
    pub(super) p: *mut CUfunc_st,
    cd: &'a CuDev,
}

unsafe impl<'a> Sync for FuncHandle<'a> {}
unsafe impl<'a> Send for FuncHandle<'a> {}

impl<'a> JITCaller<'a> {
    /// create a JITCaller from kernel image
    pub(crate) fn new(img: JITOutput<'a>) -> Result<JITCaller, cudaError_enum> {
        let mut p = null_mut();
        unsafe{cuModuleLoadData(&mut p, img.0)}
            .wrap(JITCaller { p, s: img.1, cd: img.2 })
    }
    /// get function handle by function name
    pub(crate) fn get_handle(&self, name: &str) -> Result<FuncHandle<'a>, cudaError_enum> {
        let mut p = null_mut();
        unsafe{cuModuleGetFunction(&mut p, self.p, name as *const str as *const i8)}
            .wrap(FuncHandle { p, cd: self.cd })
    }
}

impl<'a> TryFrom<JITOutput<'a>> for JITCaller<'a> {
    type Error = cudaError_enum;
    fn try_from(x: JITOutput<'a>) -> Result<JITCaller, cudaError_enum>
    { JITCaller::new(x) }
}

#[cfg(test)]
mod check_jit_caller {
    use super::*;

    #[test]
    fn new() { 
        let mut cnt = 0;
        for _ in 0..128 {
            let zk = ZooKeeper::new().unwrap();
            let cd = CuDev::new(&zk, 0, 1024).unwrap();
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
            let output = builder.build();
            println!("{output:?}");
            if output.is_err() { continue; }
            let caller : Result<JITCaller, _> = output.unwrap().try_into();
            println!("{caller:?}");
            if caller.is_err() { continue; }
            let handle = caller.unwrap().get_handle("add");
            println!("{handle:?}");
            if handle.is_err() { continue; }
            cnt += 1;
        }
        println!("{cnt:?}");
        assert!(cnt >= 32);
    }
}

/* ----------------------------------------------------------------------------------- */