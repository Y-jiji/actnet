use super::{Dev, Void};
use super::rawcuda::*;
use super::error::*;
use crate::devapi;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug)]
pub(super)
enum MsgO {
    // launch a kernel function
    RawLaunch{
        fname: String, data: Vec<*mut Void>, 
        layout: ((usize, usize, usize), (usize, usize, usize), usize)
    },
    // launch memcpy on cuda stream
    RawMemcpy{src: (*mut Void, Dev), dst: (*mut Void, Dev), len: usize},
    // set task callback on taskid
    RawTskSet{taskid: usize},
    // reply launch succeed
    RplLaunch{rplid: usize},
    // reply new box succeed
    RplNewBox{rplid: usize, boxid: usize},
    // reply del box succeed
    RplDelBox{rplid: usize},
    // reply copy box succeed
    RplCpyBox{rplid: usize, boxid: usize},
    // reply fill box succeed, rplid: usize
    RplFilBox{rplid: usize, boxid: usize},
}

unsafe impl Send for MsgO {}

pub(super)
struct ReplyPool {

}

pub(super)
struct Snder {
    rawcuda: RawCuda,
    taskpool: Arc<Mutex<TaskPool>>,
    rplypool: ReplyPool,
}

#[async_trait]
impl devapi::Snder<MsgO, DevErr> for
Snder {
    async fn snd(&mut self, msg: MsgO) -> Result<(), DevErr> {
        match msg {
            MsgO::RawLaunch { fname, data, layout } => {
                match self.rawcuda.launch(fname, data, layout) {
                    Ok(()) => Ok(()),
                    Err(raw) => Err(DevErr::Raw(raw))
                }
            }
            MsgO::RawMemcpy { src, dst, len } => {
                match self.rawcuda.memcpy(src, dst, len) {
                    Ok(()) => Ok(()),
                    Err(raw) => Err(DevErr::Raw(raw))
                }
            }
            MsgO::RawTskSet { taskid } => {
                let hook = Box::new(|| {
                    let mut lock = self.taskpool.lock();
                    lock.put(taskid);
                });
                match self.rawcuda.hookup(hook) {
                    Ok(()) => Ok(()),
                    Err(raw) => Err(DevErr::Raw(raw))
                }
            }
            x => unimplemented!("send {x:?}")
        }
    }
}