/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;

/* ----------------------------------------------------------------------------------- */

use crate::Void;
use super::err::*;
use super::keeper::*;
use super::jit::*;

/* ----------------------------------------------------------------------------------- */

use crate::raw::drv::{
    // CUDA types
    CUstream_st,
    CUstream_flags_enum,
    CUstreamCaptureMode_enum,
    // unsafe functions
    cuStreamCreate,
    cuStreamBeginCapture_v2,
    cuStreamEndCapture,
    cuGraphDestroy,
    cuGraphLaunch,
    cuGraphInstantiate_v2,
    cuGraphExecDestroy,
    cuMemcpyHtoDAsync_v2,
    cuMemcpyDtoHAsync_v2,
    cuLaunchKernel,
    cuLaunchHostFunc,
    cuStreamDestroy_v2,
    cuStreamSynchronize,
};
use std::os::raw::c_uint;
use std::mem::size_of;

/* ----------------------------------------------------------------------------------- */

pub struct CuLayout(
/// grid.x, grid.y, grid.z
    u32, u32, u32, 
/// block.x, block.y, block.z
    u32, u32, u32,
/// shared memory
    u32,
);

/* ----------------------------------------------------------------------------------- */

#[derive(Debug)]
pub struct CuStream<'a> {
    p: *mut CUstream_st,
    d: &'a CuDev,
}

unsafe extern "C" fn callback_wrapper<T>(callback: *mut Void)
where
    T: FnOnce() + Send,
{
    // Stop panics from unwinding across the FFI
    let _ = std::panic::catch_unwind(|| {
        let callback: Box<T> = Box::from_raw(callback as *mut T);
        callback();
    });
}

impl<'a> CuStream<'a> {
    /// create a new cuda stream on device
    pub fn new(d: &'a CuDev) -> Result<CuStream<'a>, cudaError_enum> {
        let mut p = null_mut();
        let flg = CUstream_flags_enum::CU_STREAM_NON_BLOCKING;
        unsafe{cuStreamCreate(&mut p, flg as c_uint)}
            .wrap(CuStream{p, d})
    }
    /// call a bundle of functions
    pub fn batch(&self, 
        f: &[&FuncHandle], l: &[&CuLayout],
        devp: &[Vec<*mut Void>],
    ) -> Result<(), cudaError_enum> {
        if f.len() != l.len() { Err(cudaError_enum::CUDA_ERROR_ASSERT)? }
        let mode = CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;
        unsafe{cuStreamBeginCapture_v2(self.p, mode)}.wrap(())?;
        for i in 0..f.len() {
            let (f, l, devp) = (f[i], l[i], devp[i].clone());
            let r = self.call(f, l, devp);
            if r.is_err() {
                let mut g = null_mut();
                unsafe{cuStreamEndCapture(self.p, &mut g).wrap(())}?;
                unsafe{cuGraphDestroy(g).wrap(())}?;
                Err(cudaError_enum::CUDA_ERROR_LAUNCH_FAILED)?
            }
        }
        let mut g = null_mut();
        unsafe{cuStreamEndCapture(self.p, &mut g).wrap(())}?;
        let mut e = null_mut();
        let r = unsafe{cuGraphInstantiate_v2(&mut e, g, null_mut(), null_mut(), 0).wrap(())};
        if r.is_err() {
            unsafe{cuGraphDestroy(g).wrap(())}?;
            Err(cudaError_enum::CUDA_ERROR_LAUNCH_FAILED)
        } else {
            let r = unsafe{cuGraphLaunch(e, self.p).wrap(())};
            if r.is_err() {
                unsafe{cuGraphExecDestroy(e).wrap(())}
            } else {
                Ok(())
            }
        }
    }
    /// copy a boxed value to computation device
    pub fn cpyd<T>(&self, v: &mut [T], p: *mut Void) -> Result<(), cudaError_enum> {
        let bc = size_of::<T>() * v.len(); // byte count for v
        let pv = (*v).as_mut_ptr();
        unsafe{cuMemcpyHtoDAsync_v2(
                p as u64, 
                pv as *mut _, 
                bc, 
                self.p
        )}.wrap(())
    }
    /// copy a value from computation device value to host
    pub fn cpyh<T>(&self, v: &mut [T], p: *mut Void) -> Result<(), cudaError_enum> {
        let bc = size_of::<T>() * v.len(); // byte count for v
        let pv = (*v).as_mut_ptr();
        unsafe{cuMemcpyDtoHAsync_v2(
            pv as *mut _, 
                p as u64, 
                bc, 
                self.p
        )}.wrap(())
    }
    /// call(f: function name, l: computation layout, p: pointers to input params)
    /// 
    /// behavior: add function execution stream to device
    /// 
    /// note: mind that if input parameter is a pointer, p should contain a pointer to this pointer
    pub fn call(&self, 
        f: &FuncHandle, l: &CuLayout, 
        p: Vec<*mut Void>, 
    ) -> Result<(), cudaError_enum> {
        let CuLayout(gx, gy, gz, bx, by, bz, shmem) = l;
        let p = p.as_ptr() as *mut _;
        unsafe{cuLaunchKernel(f.p, 
            *gx, *gy, *gz, 
            *bx, *by, *bz, 
            *shmem, self.p, p, null_mut()
        )}.wrap({drop(p); ()})
    }
    /// hook(f: the hook function)
    pub fn hook<T>(&self, h: Box<T>) -> Result<(), cudaError_enum>
        where T: FnOnce() + Send
    {
        unsafe {cuLaunchHostFunc(
            self.p,
            Some(callback_wrapper::<T>),
            Box::into_raw(h) as *mut Void,
        )}.wrap(())
    }
}

impl<'a> Drop for CuStream<'a> {
    fn drop(&mut self) {
        let mut ok = false;
        for _ in 0..5 {
            if unsafe{cuStreamSynchronize(self.p)}
                .wrap(())
                .is_ok()
            { ok = true; break }
        };
        if !ok { panic!("Cannot synchronize stream") }
        let mut ok = false;
        for _ in 0..5 {
            if unsafe{cuStreamDestroy_v2(self.p)}
                .wrap(())
                .is_ok()
            { ok = true; break }
        }
        if !ok { panic!("Cannot destroy stream") }
    }
}

/* ----------------------------------------------------------------------------------- */

#[cfg(test)]
mod check_stream {
    use super::*;

    #[test]
    fn new_and_drop() {
        // zoo keeper for devices
        let zk = ZooKeeper::new().unwrap();
        // collect handles for test threads
        let hs = (0..50).map(|_| {
            let zk = zk.clone();
            std::thread::spawn(move || {
                let cd = CuDev::new(&zk, 0, 1024).unwrap();
                let st = CuStream::new(&cd).unwrap();
                println!("{st:?}");
            })
        }).collect::<Vec<_>>();
        // join all handles
        for h in hs {h.join().unwrap();}
        // when all stream & device are dropped, zoo keeper will be empty
        assert!(zk.is_empty());
    }

    #[test]
    fn launch() {
        // zoo keeper for devices
        let zk = ZooKeeper::new().unwrap();
        let cd = CuDev::new(&zk, 0, 1024).unwrap();
        let mut builder = JITOutputBuilder::new(&cd);
        let p = include_str!("../../cu-target/test-case-1.ptx");
        // the jit builder that wraps a compiler invocation
        builder.data(
            JITInputData {
                p: p as *const _ as *mut Void, 
                t: CUjitInputType_enum::CU_JIT_INPUT_PTX, 
                s: p.len(), n: String::new()
            }
        ).unwrap();
        println!("{builder:?}");
        // the output of jit compiler
        let output = builder.build().unwrap();
        println!("{output:?}");
        // a caller where functions can be acquired
        let caller : JITCaller = output.try_into().unwrap();
        println!("{caller:?}");
        // function handle
        let fh = caller.get_handle("add").unwrap();
        println!("{fh:?}");
        // collect handles for test threads
        let hs = (0..50).map(|_| {
            let zk = zk.clone();
            let fh = fh.clone();
            std::thread::spawn(move || {
                let cd = CuDev::new(&zk, 0, 1024).unwrap();
                let st = CuStream::new(&cd).unwrap();
                todo!("copy data to cuda and call function");
                println!("{st:?}");
            })
        }).collect::<Vec<_>>();
        // join all handles
        for h in hs {h.join().unwrap();}
        // when all stream & device are dropped, zoo keeper will be empty
        drop(cd);
        assert!(zk.is_empty());
    }
}