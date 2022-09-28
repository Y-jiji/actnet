use std::fmt::{Display, Debug};
use std::net::IpAddr;

/// communicate with a device
pub trait DevAPI {
    type DevErr;
    type DevBox;
    // this design is somewhat error-prone. TODO(Y-jiji)
    type MemSeg;
    fn newbox(size: usize) -> Result<Self::DevBox, Self::DevErr>;
    fn cpybox(srcbox: &Self::DevBox) -> Result<Self::DevBox, Self::DevErr>;
    fn delbox(srcbox: Self::DevBox) -> Result<(), Self::DevErr>;
    fn filbox(data: Self::MemSeg) -> Result<Self::DevBox, Self::DevErr>;
    fn launch(name: String, data: [&mut Self::DevBox], meta: [usize]) -> Result<(), Self::DevErr>;
}

pub trait Rcver<T, E> {
    fn rcv(&mut self) -> Result<T, E>;
}

pub trait Snder<T, E> {
    fn snd(&mut self, msg:T) -> Result<(), E>;
}

/// a device as an actor
pub trait DevActor where
    Self::Snder : Snder<Self::MsgO, Self::Error>, 
    Self::Rcver : Rcver<Self::MsgI, Self::Error>,
    Self::Error : Debug
{
    /// internal state
    type State;
    /// device error
    type Error;
    /// sender
    type Snder;
    /// reciever
    type Rcver;
    /// input message
    type MsgI;
    /// output message
    type MsgO;
    /// local device api, if this actor is not net-positioned
    type LocalDevAPI;
    /// start a service with a receiver
    fn run(state: Self::State, rcver: Self::Rcver, snder: Self::Snder);
    /// start a service
    fn start(address: Option<IpAddr>) -> Self::LocalDevAPI;
}