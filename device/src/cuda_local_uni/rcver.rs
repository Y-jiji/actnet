use super::{Dev, Void};
use super::rawcuda::*;
use super::error::*;
use crate::devapi;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug)]
pub(super)
enum MsgI {
    Launch {fname: String, data: Vec<*mut Void>},
    NewBox {size: usize},
    DelBox {boxid: usize},
    FilBox {boxid: usize, data: *mut Void},
    TaskOk {taskid: usize},
    TikTok,
}

pub(super)
struct Rcver {
    taskpool: TaskPool,
}

impl Rcver {
    fn new(taskpool: TaskPool) {
    }
    fn get_rawcuda_task(&mut self) -> usize {
        self.taskpool.get()
    }
}

#[async_trait]
impl devapi::Rcver<MsgI, DevErr> for
Rcver {
    async fn rcv(&mut self) -> Result<MsgI, DevErr> {
        unimplemented!("recieve");
    }
}