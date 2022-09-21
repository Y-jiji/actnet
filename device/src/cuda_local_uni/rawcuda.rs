#![allow(warnings)]
use parking_lot::Mutex;
use std::{ptr::null_mut as nmut, marker::PhantomPinned};
use std::collections::HashMap;
use std::os::raw::c_uint;
use std::pin::Pin;
use super::{Dev, Void};

include!(concat!(env!("OUT_DIR"), "/nvtk/cuda.rs"));

pub(super)
struct RawCuda {
    pstream: *mut CUstream_st,
    pmodule: *mut CUmod_st,
    pmemseg: (*mut Void, *mut Void),
    funcmap: HashMap<String, *mut CUfunc_st>,
}

unsafe impl Send for RawCuda {}

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

impl RawCuda {
    /// a new RawCuda stream
    pub(super)
    fn new(size: usize, image: &'static str, devnr: i32) -> Result<Self, RawCudaError> {
        let mut rawcuda = Self::new_uninit();
        match RawCudaError::from(unsafe{cudaSetDevice(devnr)}) {
            RawCudaError::CUDA_SUCCESS => {},
            x => Err(x)?
        };
        rawcuda.init_stream()?;
        rawcuda.init_memory(size)?;
        rawcuda.init_module(image)?;
        return Ok(rawcuda);
    }
    /// a new uninitialized RawCuda stream
    fn new_uninit() -> Self {
        let nptr: *mut Void = nmut();
        RawCuda {
            pstream: nptr as *mut _, 
            pmodule: nptr as *mut _, 
            pmemseg: (nptr, nptr),
            funcmap: HashMap::new(),
        }
    }
    /// initialize stream
    fn init_stream(&mut self) -> Result<(), RawCudaError> {
        let errnr = unsafe{cuStreamCreate(&mut self.pstream as *mut _, 
            CUstream_flags_enum_CU_STREAM_NON_BLOCKING as u32)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => {self.pstream = nmut();Err(e)},
        }
    }
    /// initialize stream memory
    fn init_memory(&mut self, size: usize) -> Result<(), RawCudaError> {
        let errnr = unsafe{cuMemAllocAsync(
            &mut(self.pmemseg.0 as u64) as *mut _,
            size as u64, self.pstream
        )};
        match errnr {
            RawCudaError::CUDA_SUCCESS => unsafe
            {self.pmemseg.1 = self.pmemseg.0.add(size); Ok(())},
            e => 
            {self.pmemseg.0 = nmut(); Err(e)},
        }
    }
    /// initialize cuda module
    fn init_module(&mut self, image: &'static str) -> Result<(), RawCudaError> {
        let errnr  = unsafe{cuModuleLoadData(
            &mut self.pmodule as *mut _,
            image as *const _ as *mut _,
        )};
        match errnr {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => 
            {self.pmodule = nmut(); Err(e)},
        }
    }
    /// get copy kind from device type
    #[inline]
    fn cpyknd(src: Dev, dst: Dev) -> i32 {
        match (src, dst) {
            (Dev::Host, Dev::Host) => cudaMemcpyKind_cudaMemcpyHostToHost,
            (Dev::Host, Dev::Cuda) => cudaMemcpyKind_cudaMemcpyHostToHost,
            (Dev::Cuda, Dev::Host) => cudaMemcpyKind_cudaMemcpyDeviceToHost,
            (Dev::Cuda, Dev::Cuda) => cudaMemcpyKind_cudaMemcpyDeviceToDevice,
        }
    }
    /// add a memory copy job to stream
    pub(super)
    fn memcpy(&mut self, src: (*mut Void, Dev), dst: (*mut Void, Dev), len: usize) -> Result<(), RawCudaError> {
        let errnr = unsafe{cudaMemcpyAsync(
            dst.0, src.0, len as _, 
            Self::cpyknd(src.1, dst.1) as _, self.pstream)};
        match RawCudaError::from(errnr) {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
    /// add a kernel function launch job to stream
    pub(super)
    fn launch(
        &mut self, fname: String, data: Vec<*mut Void>, 
        layout: ((usize, usize, usize), (usize, usize, usize), usize)
    ) -> Result<(), RawCudaError> {
        let pstream = self.pstream;
        let pfunc = match self.funcmap.get(&fname) {
            Some(pfunc) => pfunc.clone(),
            None => Err(RawCudaError::CUDA_ERROR_NOT_FOUND)?,
        };
        let errnr = unsafe{cuLaunchKernel(pfunc, 
            layout.0.0 as c_uint, layout.0.1 as c_uint, layout.0.2 as c_uint, 
            layout.1.0 as c_uint, layout.1.1 as c_uint, layout.1.2 as c_uint, 
            layout.2 as c_uint, pstream, 
            data.as_ptr() as *mut *mut Void, nmut())};
        match errnr {
            RawCudaError::CUDA_SUCCESS => Ok(()),
            e => Err(e),
        }
    }
    /// add a host hook that will launch when current jobs are done
    pub(super)
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

impl Drop for RawCuda {
    fn drop(&mut self) {
        todo!("Drop RawCuda");
    }
}

/// a communication model for cuda stream to send state to host
pub(super)
struct TaskPool {
    /// 1K Bytes for signal slots
    signal: [u8; 1024],
    /// start offset
    offset: usize,
    /// used length,
    uselen: usize,
    /// signal should be pinned in memory
    pinpin: PhantomPinned,
}

impl TaskPool {
    fn new() -> Self {
        // unimplemented!("new TaskPool");
        TaskPool {
            signal: [0u8; 1024],
            offset: 0, uselen: 0, 
            pinpin: PhantomPinned, 
        }
    }
    /// get an empty task
    pub(super)
    fn get(&mut self) -> usize {
        if self.uselen == 1024*8 { return 1024*8; }
        let task = (self.offset + self.uselen) & (1024*8-1);
        self.uselen += 1;
        return task;
    }
    /// set self.signal[task] to 1
    pub(super)
    fn put(&mut self, task: usize) {
        self.signal[task >> 3] |= 1 << (task & 0x7);
    }
    /// return the finished tasks, put these task bits back to pool
    pub(super)
    fn ack(&mut self) -> Vec<usize> {
        let mut ret = Vec::with_capacity(16);
        while self.signal[self.offset >> 3] == u8::MAX {
            self.signal[self.offset >> 3] = 0u8;
            ret.extend(self.offset..self.offset+8);
            self.offset += 8;
            self.offset &= 1024*8 - 1;
            self.uselen -= 8;
        }
        while self.signal[self.offset >> 3] & (1 << (self.offset & 0x7)) != 0 {
            self.signal[self.offset >> 3] ^= (1 << (self.offset & 0x7));
            ret.push(self.offset);
            self.offset += 1;
            self.offset &= 1024*8 - 1;
            self.uselen -= 1;
        }
        ret.shrink_to_fit();
        return ret;
    }
    /// return whether the task pool is 2/3 full
    pub(super)
    fn is_full(&self) -> bool {
        self.uselen >= 1024*6
    }
    /// print the task pool
    #[cfg(test)]
    pub(super)
    fn print(&self) {
        let each_row = 16;
        let sep = "=".repeat(each_row*8 + each_row-1);
        println!("{sep}");
        for i in 0..(1024/each_row) {
            for j in 0..each_row {
                for k in 0..8 {
                    print!("{}", (self.signal[i*each_row+j] >> k) & 1);
                }
                print!(" ");
            }
            print!("\n");
        }
        println!("{sep}");
    }
}

#[cfg(test)]
mod test {
    use super::TaskPool;
    #[test]
    fn test_task_pool() {
        let mut x = TaskPool::new();
        x.print();
        for _ in 0..2019 {
            for _ in 0..2109 {
                let tid = x.get();
                x.put(tid);
            }
            x.ack();
        }
        for _ in 0..20 {
            for _ in 0..7 {
                let tid = x.get();
                x.put(tid);
                x.print();
            }
            x.ack();
            x.print();
        }
        for _ in 0..(1024*8-1) {x.get();}
        println!("{}", x.get());
        println!("{}", x.get());
    }
}