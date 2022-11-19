/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;

/* ----------------------------------------------------------------------------------- */

use crate::Void;
use super::err::*;
use super::zk::*;
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
    cd: &'a CuDev,
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
    pub fn new(cd: &'a CuDev) -> Result<CuStream<'a>, cudaError_enum> {
        let mut p = null_mut();
        let flg = CUstream_flags_enum::CU_STREAM_NON_BLOCKING;
        unsafe{cuStreamCreate(&mut p, flg as c_uint)}
            .wrap(CuStream{p, cd})
    }
    /// call a bundle of functions
    pub unsafe fn batch(&self, 
        f: &[&FuncHandle], l: &[&CuLayout],
        devp: &[Vec<*mut Void>],
    ) -> Result<(), cudaError_enum> {
        if f.len() != l.len() { Err(cudaError_enum::CUDA_ERROR_ASSERT)? }
        let mode = CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED;
        cuStreamBeginCapture_v2(self.p, mode).wrap(())?;
        for i in 0..f.len() {
            let (f, l, devp) = (f[i], l[i], devp[i].clone());
            let r = self.call(f, l, devp);
            if r.is_err() {
                let mut g = null_mut();
                cuStreamEndCapture(self.p, &mut g).wrap(())?;
                cuGraphDestroy(g).wrap(())?;
                Err(cudaError_enum::CUDA_ERROR_LAUNCH_FAILED)?
            }
        }
        let mut g = null_mut();
        cuStreamEndCapture(self.p, &mut g).wrap(())?;
        let mut e = null_mut();
        let r = cuGraphInstantiate_v2(&mut e, g, null_mut(), null_mut(), 0).wrap(());
        if r.is_err() {
            cuGraphDestroy(g).wrap(())?;
            Err(cudaError_enum::CUDA_ERROR_LAUNCH_FAILED)
        } else {
            let r = cuGraphLaunch(e, self.p).wrap(());
            if r.is_err() {
                cuGraphExecDestroy(e).wrap(())
            } else {
                Ok(())
            }
        }
    }
    /// copy a boxed value to computation device
    pub fn cpyd<T>(&self, v: &mut [T], p: &*mut Void) -> Result<(), cudaError_enum> {
        let bc = size_of::<T>() * v.len(); // byte count for v
        let pv = (*v).as_mut_ptr();
        unsafe{cuMemcpyHtoDAsync_v2(
                *p as u64, 
                pv as *mut _, 
                bc, 
                self.p
        )}.wrap(())
    }
    /// copy a value from computation device value to host
    pub fn cpyh<T>(&self, v: &mut [T], p: &*mut Void) -> Result<(), cudaError_enum> {
        let bc = size_of::<T>() * v.len(); // byte count for v
        let pv = (*v).as_mut_ptr();
        unsafe{cuMemcpyDtoHAsync_v2(
            pv as *mut _, 
                *p as u64, 
                bc, 
                self.p
        )}.wrap(())
    }
    /// call(f: function name, l: computation layout, p: pointers to input params)
    /// 
    /// behavior: add function execution stream to device
    /// 
    /// note: mind that if input parameter is a pointer, p should contain a pointer to this pointer
    pub unsafe fn call(&self, 
        f: &FuncHandle, l: &CuLayout, 
        p: Vec<*mut Void>, 
    ) -> Result<(), cudaError_enum> {
        let CuLayout(gx, gy, gz, bx, by, bz, shmem) = l;
        let p = p.as_ptr() as *mut _;
        cuLaunchKernel(f.p, 
            *gx, *gy, *gz, 
            *bx, *by, *bz, 
            *shmem, self.p, p, null_mut()
        ).wrap({drop(p); ()})
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
    use rand::*;

    #[test]
    fn new_and_drop() {
        // zoo keeper for devices
        let zk = ZooKeeper::new().unwrap();
        // collect handles for test threads
        let hs = (0..8).map(|_| {
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
        let cd = CuDev::new(&zk, 0, 3*size_of::<i32>()*256*32).unwrap();
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
        for i in 0..32 {unsafe {
            // initialize a stream
            let st = CuStream::new(&cd).unwrap();
            // initialize test data for i32 add
            let mut vx = [1i32; 256];
            for j in 0..256 {vx[j] = random::<i32>() % 1024;}
            let mut vy = [2i32; 256];
            for j in 0..256 {vy[j] = random::<i32>() % 1024;}
            let mut vz = [0i32; 256];
            for j in 0..256 {vz[j] = random::<i32>() % 1024;}
            // move data to device pointers
            let mut px = cd.p.add((0+3*i)*256*size_of::<i32>());
            let mut py = cd.p.add((1+3*i)*256*size_of::<i32>());
            let mut pz = cd.p.add((2+3*i)*256*size_of::<i32>());
            st.cpyd(&mut vx, &px).unwrap();
            st.cpyd(&mut vy, &py).unwrap();
            st.cpyd(&mut vz, &pz).unwrap();
            // specify test layout
            let l = CuLayout(1, 1, 1, 8, 8, 8, 0);
            let mut len = 256i32;
            let param: Vec<*mut Void> = vec![
                &mut px as *mut _ as *mut Void, 
                &mut py as *mut _ as *mut Void, 
                &mut pz as *mut _ as *mut Void,
                &mut len as *mut _ as *mut Void,
            ];
            // call "add" by function handle
            st.call(&fh, &l, param).unwrap();
            // copy back values
            st.cpyh(&mut vx, &px).unwrap();
            st.cpyh(&mut vy, &py).unwrap();
            st.cpyh(&mut vz, &pz).unwrap();
            // drop stream to synchronize all operations
            drop(st);
            // assert call has right side-effects
            for j in 0..256 { assert!(vx[j] + vy[j] == vz[j]) }
        }};
        // drop the cuda device
        drop(cd);
        // when all stream & device are dropped, zoo keeper will be empty
        assert!(zk.is_empty());
    }
}