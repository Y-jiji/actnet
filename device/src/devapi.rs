use std::fmt::{Display, Debug};



/// communicate with a device
pub trait DevAPI {
    type DevErr;
    type DevBox;
    // this design is somewhat error-prone. TODO(Y-jiji)
    type MemSeg;
    fn newbox(size: usize) -> Result<Self::DevBox, Self::DevErr>;
    fn cpybox(srcbox: &Self::DevBox) -> Result<Self::DevBox, Self::DevErr>;
    fn delbox(srcbox: Self::DevBox) -> Result<(), Self::DevErr>;
    fn filbox(dstbox: Self::DevBox, data: Self::MemSeg) -> Result<(), Self::DevErr>;
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
    /// TODO(Y-jiji): Add DevAPI as a type, return DevAPI after start
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
    /// start a service with a receiver
    fn run(state: Self::State, rcver: Self::Rcver, snder: Self::Snder);
    /// start a service
    fn start(self);
}