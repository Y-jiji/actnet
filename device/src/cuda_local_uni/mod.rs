mod rawcuda;
mod snder;
mod rcver;
mod state;
mod error;
use super::devapi::*;
use async_trait::async_trait;

type Void = std::ffi::c_void;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Dev { Cuda=1, Host=0 }

#[cfg(test)]
/// tests for cuda_local_uni
mod tests;

struct CudaLocalUni {
    state: state::ShadowMem,
    rcver: rcver::Rcver,
    snder: snder::Snder,
}

#[async_trait]
impl DevActor
for CudaLocalUni {
    type Error = error::DevErr;
    type MsgI = rcver::MsgI;
    type MsgO = snder::MsgO;
    type Snder = snder::Snder;
    type Rcver = rcver::Rcver;
    type State = state::ShadowMem;
    async fn run(state: Self::State, mut rcver: Self::Rcver, mut snder: Self::Snder) {loop {
        let msgi = match rcver.rcv().await {
            Err(e) => panic!("{e:?}"),
            Ok(msgi) => msgi,
        };
        match msgi {
            rcver::MsgI::TaskOk{taskid} => {
            },
            msgi => {
                todo!("{msgi:?}");
            }
        }
    }}
    async fn start(self) {
        Self::run(self.state, self.rcver, self.snder).await;
    }
}