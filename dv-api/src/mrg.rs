//! traits for cooperative functions

use crate::*;

/// an error that happens
pub enum Either<A, B> {A(A), B(B)}

/// data copy between peer devices
pub trait PeerCopy<D0: Device, D1: Device>
where Self: Device {
    /// default implementation is [d0 -> host -> d1]
    fn pcpy(d0: &D0, d1: &D1, s0: &D0::Symbol, s1: &mut D1::Symbol)
        -> Result<(), (ComErr, Either<D0::DevErr, D1::DevErr>)> 
    {
        let datvec: WrapVec = match d0.dump(s0) {
            Err((ce, de)) => Err((ce, Either::A(de)))?,
            Ok(datbox) => datbox.as_vec(),
        };
        match d1.load(D1::DatBox::from_vec(datvec), s1) {
            Err((ce, de)) => Err((ce, Either::B(de)))?,
            Ok(s1) => Ok(s1),
        }
    }
}