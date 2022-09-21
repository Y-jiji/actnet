use super::{Dev, Void};
use super::rawcuda::*;
use super::error::*;
use crate::devapi;

use std::sync::Arc;
use std::sync::mpsc::Receiver;
use parking_lot::Mutex;
use std::thread;
use std::time::{Instant, Duration};

#[derive(Debug)]
pub(super)
enum MsgI {
    // operation launch an operator
    Launch {fname: String, data: Vec<*mut Void>},
    // operation: create a new box
    NewBox {size: usize},
    // operation: delete box
    DelBox {boxid: usize},
    // operation: fill box
    FilBox {boxid: usize, data: *mut Void},
    // list of ok task
    TaskOk {lstid: Vec<usize>},
}

unsafe impl Send for MsgI {}

pub(super)
struct TimerCheck {
    timercv: Receiver<()>,
    dur: u64
}

impl TimerCheck {
    fn new(dur: u64) -> Self {
        assert!(dur << 3 > 0);
        let (snd, rcv) = std::sync::mpsc::sync_channel(1);
        thread::spawn(move ||{
            let mut start = Instant::now();
            loop {
                let duration = start.elapsed();
                if duration > Duration::from_millis(dur) {
                    snd.send(()).unwrap();
                    start = Instant::now();
                }
                if (Duration::from_millis(dur) - duration) > Duration::from_millis(20) {
                    thread::sleep((Duration::from_millis(dur) - duration)/2);
                }
            }
        });
        TimerCheck { timercv: rcv, dur }
    }
    fn rcv(&self) -> Option<()> {
        match self.timercv.recv_timeout(std::time::Duration::from_millis(self.dur >> 3)) {
            Ok(()) => Some(()),
            Err(_) => None,
        }
    }
}

pub(super)
struct Rcver {
    // protected reference to task pool
    taskpool: Arc<Mutex<TaskPool>>,
    // timer
    timerchk: TimerCheck,
    // receiver from caller
    loclrecv: Receiver<MsgI>,
}

impl Rcver {
    const DUR : u64 = 1 << 7;
    fn new(taskpool: Arc<Mutex<TaskPool>>, loclrecv: Receiver<MsgI>) -> Self {
        let timerchk = TimerCheck::new(Self::DUR);
        Rcver { taskpool, timerchk, loclrecv }
    }
}


impl devapi::Rcver<MsgI, DevErr> for
Rcver {
    fn rcv(&mut self) -> Result<MsgI, DevErr> {
        // timer signal arrived, receive finished tasks
        if self.timerchk.rcv() == Some(()) || self.taskpool.lock().is_full()
        { return Ok(MsgI::TaskOk { lstid: self.taskpool.lock().ack() }); }
        // a locally sent a message
        match self.loclrecv.recv() {
            Ok(msg) => Ok(msg),
            Err(e) => Err(DevErr::RecvError),
        }
    }
}