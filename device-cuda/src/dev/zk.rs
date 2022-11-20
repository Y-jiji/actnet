/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;
use std::thread;
use std::thread::ThreadId;
use std::sync::Arc;

/* ----------------------------------------------------------------------------------- */

use super::err::*;
use scc::HashSet;
use crate::Void;

/* ----------------------------------------------------------------------------------- */
use raw::drv::{
    // CUDA types
    CUdevice,
    CUctx_st,
    CUctx_flags_enum,
    // unsafe functions
    cuInit,
    cuCtxCreate_v2,
    cuCtxDestroy_v2,
    cuDeviceGet,
    cuMemAlloc_v2,
    cuMemFree_v2,
};
use std::os::raw::c_int;

/* ----------------------------------------------------------------------------------- */

/// record whether a thread is operating on a device
/// 
/// a ZooKeeper should be created per-process before other objects
#[derive(Clone, Debug)]
pub struct ZooKeeper(Arc<HashSet<ThreadId>>);

impl ZooKeeper {
    pub fn new() -> Result<ZooKeeper, cudaError_enum> {
        unsafe{cuInit(0)}.wrap(())?;
        Ok(ZooKeeper(Arc::new(HashSet::new())))
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// a cuda device
/// 
/// each thread is guaranteed to have a unique CudaDev object if only one ZooKeeper is used in this computer
#[derive(Debug)]
pub struct CuDev {
    /// cuda context
    pub(super) c: *mut CUctx_st,
    /// device number
    pub n: CUdevice,
    /// memory base
    pub(super) p: *mut Void,
    /// memory size
    pub(super) s: usize,
    /// a global zoo keeper (daemon object?) to keep uniqueness of device on each thread
    zk: ZooKeeper,
}

impl CuDev {
    /// initialize a new CudaDev if there is no CudaDev on this thread
    /// else, return CUDA_ERROR_INVALID_DEVICE
    pub fn new(zk: &ZooKeeper, n: usize, s: usize) -> Result<CuDev, cudaError_enum> {
        let tid = thread::current().id();
        let zk = zk.clone();
        // query the device
        match zk.0.read(&tid, |_| true) {
            None => unsafe {
                let mut c = null_mut();
                let flg = CUctx_flags_enum::CU_CTX_BLOCKING_SYNC as u32;
                let n = {
                    let mut d = -1i32; 
                    cuDeviceGet(&mut d, n as c_int).wrap(())?; d
                };
                cuCtxCreate_v2(&mut c, flg, n).wrap(())?;
                zk.0.insert(tid).expect("This should not happen, each key have only one thread");
                let p = {
                    let mut p = 0u64;
                    cuMemAlloc_v2(&mut p, s);
                    null_mut::<Void>().add(p as usize)
                };
                Ok(CuDev {c, n, zk, p, s})
            }
            Some(_) => Err(cudaError_enum::CUDA_ERROR_INVALID_DEVICE),
        }
    }
}

impl Drop for CuDev {
    fn drop(&mut self) {
        let tid = thread::current().id();
        let r = unsafe{cuMemFree_v2(self.p as u64).wrap(())};
        debug_assert!(r.is_ok(), "cuMemFree failed");
        let r = unsafe{cuCtxDestroy_v2(self.c)}.wrap(());
        debug_assert!(r.is_ok(), "cuCtxDestroy failed");
        let ZooKeeper(inner) = &self.zk;
        inner
            .remove(&tid)
            .expect("A device is in use but unrecorded on thread {tid:?}. ");
    }
}

/* ----------------------------------------------------------------------------------- */

#[cfg(test)]
mod check_zoo_keeper {
    use super::*;

    #[test]
    fn new_and_drop() {
        let zk = ZooKeeper::new().expect("cuInit(0) failed");
        let zk_1 = zk.clone();
        let handle_1 = std::thread::spawn(move || {
            let cuda_device_1 = CuDev::new(&zk_1, 0, 1024).unwrap();
            let cuda_device_2 = CuDev::new(&zk_1, 0, 1024).expect_err("This should be an error");
            drop(zk_1);
            drop(cuda_device_1);
            drop(cuda_device_2);
        });
        let zk_2 = zk.clone();
        let handle_2 = std::thread::spawn(move || {
            let cuda_device_1 = CuDev::new(&zk_2, 0, 1024).unwrap();
            let cuda_device_2 = CuDev::new(&zk_2, 0, 1024).expect_err("This should be an error");
            drop(zk_2);
            drop(cuda_device_1);
            drop(cuda_device_2);
        });
        handle_1.join().unwrap();
        handle_2.join().unwrap();
    }
}

/* ----------------------------------------------------------------------------------- */