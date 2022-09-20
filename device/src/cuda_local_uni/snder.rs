// #![allow(warnings)]
use super::{Dev, Void};
use super::rawcuda::*;
use super::error::*;
use crate::devapi;
use async_trait::async_trait;

#[derive(Debug)]
pub(super)
enum MsgO {
    RawLaunch{
        fname: String, data: Vec<*mut Void>, 
        layout: ((usize, usize, usize), (usize, usize, usize), usize)
    },
    // launch memcpy on cuda stream
    RawMemcpy{
        src: (*mut Void, Dev), 
        dst: (*mut Void, Dev)
    },
    // set task callback on taskid
    RawTskSet{
        taskid: usize
    },
    // reply launch succeed
    RplLaunch{
        // reply to whom
        rplid: usize, 
    },
    // reply new box succeed
    RplNewBox{
        // reply to whom
        rplid: usize, 
    },
    // reply del box succeed
    RplDelBox{
        // reply to whom
        rplid: usize,
    },
    // reply copy box succeed
    RplCpyBox{
        // reply to whom
        rplid: usize
    },
    // reply fill box succeed
    RplFilBox{
        // reply to whom
        rplid: usize
    },
}

unsafe impl Send for MsgO {}

pub(super)
struct Snder {
}

#[async_trait]
impl devapi::Snder<MsgO, DevErr> for
Snder {
    async fn snd(&mut self, msg: MsgO) -> Result<(), DevErr> {
        match msg {
            x => unimplemented!("send {x:?}")
        }
        Ok(())
    }
}