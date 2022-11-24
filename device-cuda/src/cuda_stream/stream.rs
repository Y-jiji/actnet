use crate::cuda_wrap::*;
use device_api::*;
use ouroboros::self_referencing;
use mem_allocator::MemShadow;
use std::rc::Rc;
use std::cell::RefCell;

#[self_referencing]
/// self-referencing struct that manages a stream and correspondent device
pub struct InnerCudaStream {
    cd: CuDev,
    #[borrows(cd)]
    #[not_covariant]
    cs: CuStream<'this>,
}

/// a very simple implementation of CudaStream
pub struct CudaStream {
    pub(super)
    zk: ZooKeeper,
    pub(super)
    inner: InnerCudaStream,
    pub(super)
    mm: Rc<RefCell<MemShadow<8>>>,
}

const KB: usize = 1024;
const MB: usize = 1024*KB;
const GB: usize = 1024*MB;

impl CudaStream {
    /// create a new cuda stream with given size on default device (device 0)
    pub fn new(s: usize) -> Result<CudaStream, (ComErr, cudaError_enum)> {
        let zk = match ZooKeeper::new() {
            Err(e) => Err((ComErr::InitFailure, e))?,
            Ok(zk) => zk,
        };
        let cd = match CuDev::new(&zk, 0, s) {
            Err(e) => Err((ComErr::InitFailure, e))?,
            Ok(cd) => cd,
        };
        let inner = match (InnerCudaStreamTryBuilder {
            cd, cs_builder: |cd| CuStream::new(cd)
        }).try_build() {
            Err(e) => Err((ComErr::InitFailure, e))?,
            Ok(inner) => inner, 
        };
        let msize = inner.borrow_cd().s;
        let mbase = inner.borrow_cd().p;
        let level = vec![256, 1*KB, 4*KB, 16*KB, 4*MB, 16*MB, 128*MB, GB];
        let mm = Rc::new(RefCell::new(MemShadow::new(msize, mbase, level)));
        Ok(CudaStream { zk, inner, mm })
    }
}