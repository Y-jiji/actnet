use super::devapi::*;
use std::sync::mpsc::Sender;

type Void = std::ffi::c_void;

pub enum DevCli<D: DevAPI> {
    Local {
        sender: Sender<DevMsg<D>>
    },
    Inet {
        /* todo */
    }
}

pub enum DevMsg<D: DevAPI> {
    Launch {function: String, data: Vec<D::DevBox>, meta_u: Vec<usize>, meta_f: Vec<f32>, reply_to: D::ReplySend},
    NewBox {size: usize, reply_to: D::ReplySend},
    DelBox {devbox: D::DevBox, reply_to: D::ReplySend},
    ClnBox {devbox: D::DevBox, reply_to: D::ReplySend},
    PutBox {dstbox: D::DevBox, src: *mut Void, len: usize, reply_to: D::ReplySend},
    GetBox {srcbox: D::DevBox, dst: *mut Void, len: usize, reply_to: D::ReplySend},
    Sync,
}

pub(crate)
trait ReplyReceiver<D: DevAPI> {
    fn recv(self) -> Result<DevReply<D>, D::DevErr>;
}

trait DevCliModal<D: DevAPI>
where Self::ReplyRecv: ReplyReceiver<D>,
{
    type ReplyRecv;
    fn channel(&mut self) -> (D::ReplySend, Self::ReplyRecv);
    fn send(&mut self, msg: DevMsg<D>) -> Result<(), D::DevErr>;
    fn launch(&mut self, function: String, data: Vec<D::DevBox>, meta_u: Vec<usize>, meta_f: Vec<f32>) -> Result<(), D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::Launch{function, data, meta_u, meta_f, reply_to: replysend})?;
        replyrecv.recv()?.to_void()
    }
    fn newbox(&mut self, size: usize) -> Result<D::DevBox, D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::NewBox{size, reply_to: replysend})?;
        replyrecv.recv()?.to_devbox()
    }
    fn delbox(&mut self, devbox: D::DevBox) -> Result<(), D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::DelBox { devbox, reply_to: replysend })?;
        replyrecv.recv()?.to_void()
    }
    fn clnbox(&mut self, devbox: D::DevBox) -> Result<D::DevBox, D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::ClnBox { devbox, reply_to: replysend })?;
        replyrecv.recv()?.to_devbox()
    }
    fn putbox(&mut self, dstbox: D::DevBox, src: *mut Void, len: usize) -> Result<(), D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::PutBox {dstbox, src, len, reply_to: replysend})?;
        replyrecv.recv()?.to_void()
    }
    fn getbox(&mut self, srcbox: D::DevBox, dst: *mut Void, len: usize) -> Result<(), D::DevErr> {
        let (replysend, replyrecv) = self.channel();
        self.send(DevMsg::<D>::GetBox {srcbox, dst, len, reply_to: replysend})?;
        replyrecv.recv()?.to_void()
    }
    fn sync(&mut self) -> Result<(), D::DevErr> {
        self.send(DevMsg::Sync)
    }
}