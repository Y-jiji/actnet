/* ----------------------------------------------------------------------------------- */

use std::ptr::null_mut;
use std::thread;
use std::thread::ThreadId;
use std::sync::Arc;

/* ----------------------------------------------------------------------------------- */

use super::err::*;
use scc::HashSet;

/* ----------------------------------------------------------------------------------- */

use crate::raw::drv::{
    // CUDA types
    CUdevice,
    CUctx_st,
    CUctx_flags_enum,
    // unsafe functions
    cuInit,
    cuCtxCreate_v2,
    cuCtxDestroy_v2,
    cuDeviceGet,
};
use std::os::raw::c_int;

/* ----------------------------------------------------------------------------------- */

/// record whether a thread is operating on a device
/// 
/// a ZooKeeper should be created per-process before other works
#[derive(Clone, Debug)]
pub struct ZooKeeper(Arc<HashSet<ThreadId>>);

impl ZooKeeper { 
    pub fn new() -> Result<ZooKeeper, cudaError_enum> {
        unsafe{cuInit(0)}.wrap(())?;
        Ok(ZooKeeper(Arc::new(HashSet::new())))
    } 
}

/// a cuda device
/// 
/// each thread is guaranteed to have a unique CudaDev object if only one ZooKeeper is used in this computer
#[derive(Debug)]
pub struct CudaDev {
    /// cuda context
    pub(super) c: *mut CUctx_st,
    /// device number
    pub n: CUdevice,
    /// a global keeper for each device
    k: ZooKeeper,
}

impl CudaDev {
    /// initialize a new CudaDev if there is no CudaDev on this thread
    /// else, return CUDA_ERROR_INVALID_DEVICE
    pub fn new(zk: &ZooKeeper, n: usize) -> Result<CudaDev, cudaError_enum> {
        let tid = thread::current().id();
        let k = zk.clone();
        // query the device
        match k.0.read(&tid, |_| true) {
            None => unsafe {
                let mut c = null_mut();
                let flg = CUctx_flags_enum::CU_CTX_BLOCKING_SYNC as u32;
                let n = {
                    let mut d = -1i32; 
                    cuDeviceGet(&mut d, n as c_int).wrap(())?; d
                };
                cuCtxCreate_v2(&mut c, flg, n).wrap(())?;
                k.0.insert(tid).expect("This should not happen, each key have only one thread");
                Ok(CudaDev {c, n, k})
            }
            Some(_) => Err(cudaError_enum::CUDA_ERROR_INVALID_DEVICE),
        }
    }
}

impl Drop for CudaDev {
    fn drop(&mut self) {
        let tid = thread::current().id();
        let r = unsafe{cuCtxDestroy_v2(self.c)}.wrap(());
        debug_assert!(r.is_ok(), "cuCtxDestroy failed");
        let ZooKeeper(inner) = &self.k;
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
        let keeper = ZooKeeper::new().expect("cuInit(0) failed");
        let keeper_1 = keeper.clone();
        let handle_1 = std::thread::spawn(move || {
            let cuda_device_1 = CudaDev::new(&keeper_1, 0).unwrap();
            let cuda_device_2 = CudaDev::new(&keeper_1, 0).expect_err("This should be an error");
            drop(keeper_1);
            drop(cuda_device_1);
            drop(cuda_device_2);
        });
        let keeper_2 = keeper.clone();
        let handle_2 = std::thread::spawn(move || {
            let cuda_device_1 = CudaDev::new(&keeper_2, 0).unwrap();
            let cuda_device_2 = CudaDev::new(&keeper_2, 0).expect_err("This should be an error");
            drop(keeper_2);
            drop(cuda_device_1);
            drop(cuda_device_2);
        });
        handle_1.join().unwrap();
        handle_2.join().unwrap();
    }
}

/* ----------------------------------------------------------------------------------- */