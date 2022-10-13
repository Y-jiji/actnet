#![allow(warnings)]
use parking_lot::Mutex;
use std::{ptr::null_mut as nmut, marker::PhantomPinned};
use std::collections::HashMap;
use std::os::raw::c_uint;
use std::pin::Pin;
use std::ptr::*;
use super::*;

pub(crate)
enum DPtr<PtrT> {
    Host(PtrT),
    Device(PtrT),
}

type Void = std::ffi::c_void;

/// a minimal wrapper over CUDAStream
#[derive(Debug)]
pub(crate)
struct RawStream {
    pstream: *mut CUstream_st,
    devnr: i32,
}

/// a small utility to call host function indirectly
unsafe extern "C"
fn callback_wrapper<T>(callback: *mut Void)
where
    T: FnOnce() + Send,
{
    // Stop panics from unwinding across the FFI
    let _ = std::panic::catch_unwind(|| {
        let callback: Box<T> = Box::from_raw(callback as *mut T);
        callback();
    });
}

impl RawStream {
    /// a new RawCuda stream
    pub(crate)
    fn new() -> Result<Self, RawCudaError> {
        Ok(RawStream {
            pstream: Self::init_pstream()?, 
            devnr: Self::init_devnr()?,
        })
    }
    /// get device number (for the sake of sanity check)
    pub(crate)
    fn init_devnr() -> Result<i32, RawCudaError> {
        let mut devnr: i32 = -1;
        let errnr = unsafe{cudaGetDevice(&mut devnr as *mut i32)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => Ok(devnr),
            e => {Err(e)}
        }
    }
    /// initialize stream
    pub(crate)
    fn init_pstream() -> Result<CUstream, RawCudaError> {
        let mut pstream = null_mut::<CUstream_st>();
        let errnr = unsafe{cuStreamCreate(&mut pstream as *mut CUstream, 
            CUstream_flags_enum_CU_STREAM_NON_BLOCKING as u32)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => Ok(pstream),
            e => Err(e),
        }
    }
    /// get copy kind from device type
    #[inline]
    fn cpyknd<PtrT>(src: DPtr<PtrT>, dst: DPtr<PtrT>) -> (i32, PtrT, PtrT) {
        match (src, dst) {
            (DPtr::Host(s), DPtr::Host(d)) => (cudaMemcpyKind_cudaMemcpyHostToHost, s, d),
            (DPtr::Host(s), DPtr::Device(d)) => (cudaMemcpyKind_cudaMemcpyHostToHost, s, d),
            (DPtr::Device(s), DPtr::Host(d)) => (cudaMemcpyKind_cudaMemcpyDeviceToHost, s, d),
            (DPtr::Device(s), DPtr::Device(d)) => (cudaMemcpyKind_cudaMemcpyDeviceToDevice, s, d)
        }
    }
    /// add a memory copy job to stream
    pub(crate)
    fn memcpy(&mut self, src: DPtr<*mut Void>, dst: DPtr<*mut Void>, len: usize) -> Result<(), RawCudaError> {
        let (kind, src, dst) = Self::cpyknd(src, dst);
        let count = len as u64;
        let errnr = unsafe{cudaMemcpyAsync(dst, src, count, kind, self.pstream)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
    /// add a kernel function launch job to stream
    /// 
    /// pfunc: a device pointer to device function
    /// 
    /// layout: memory layout (gridx, gridy, gridz) (blockx, blocky, blockz) shardedMemBytes
    /// 
    /// data: pointers to data in host, should be suffixed by nullptrs, 16 at most
    pub(crate)
    fn launch(
        &mut self, pfunc: *mut Void, 
        layout: ((u32, u32, u32), (u32, u32, u32), u32), 
        data: Pin<Vec<*mut Void>>
    ) -> Result<(), RawCudaError> {
        let hstream = self.pstream;
        let ((gridDimX, gridDimY, gridDimz), (blockDimX, blockDimY, blockDimZ), sharedMemBytes) = layout;
        let kernelParams = data.as_ptr() as *mut *mut Void;
        let errnr = unsafe{cuLaunchKernel(
            pfunc as *mut _, 
            gridDimX, gridDimY, gridDimz,
            blockDimX, blockDimY, blockDimZ, sharedMemBytes, 
            hstream, kernelParams, nmut())};
        match errnr {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
    /// add a host hook that will launch when current jobs are done
    pub(crate)
    fn hookup<T>(&mut self, hook: Box<T>) -> Result<(), RawCudaError>
    where T: FnOnce() + Send {
        let errnr = unsafe{cuLaunchHostFunc(
            self.pstream,
            Some(callback_wrapper::<T>),
            Box::into_raw(hook) as *mut Void,
        )};
        match errnr {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
}

impl Drop for RawStream {
    fn drop(&mut self) {
        let errnr = unsafe{cudaStreamDestroy(self.pstream)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => {},
            error => panic!("{:?} occurs when dropping {:?} on CUDA device({:?})", error, self.pstream, self.devnr),
        }
        drop(self.devnr);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn raw_stream_init() {
        let raw_stream = RawStream::new();
        println!("{raw_stream:?}");
        drop(raw_stream);
    }
}