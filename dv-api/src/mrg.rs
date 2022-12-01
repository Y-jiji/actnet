//! traits for cooperative functions

use crate::*;

#[derive(Debug, Clone)]
pub enum Either<A, B> {A(A), B(B)}

/// ### *description*
/// peer copy from self to other device
pub trait PeerCopyTo<Other: Device> 
where Self: Device {
    /// ### *description*
    /// peer copy to *`dst`* on *`other`* from *`src`* symbol on *`self`* 
    fn prcpy(&self, other: &Other, src: &Self::Symbol, dst: &mut Other::Symbol)
        -> Result<(), Either<DevErr<Self>, DevErr<Other>>>
    {
        //! ### *default implementation*
        //! *`other`* --> *`main memory`* --> *`self`*
        other.load(Other::DatBox::from_vec(self.dump(src).map_err(Either::A)?.as_vec()), dst).map_err(Either::B)
    }
}

/// ### *description*
/// peer copy from other device to self
pub trait PeerCopyFrom<Other: Device>
where Self: Device {
    /// ### *description*
    /// peer copy to *`dst`* on *`self`* from *`src`* symbol on *`other`*
    fn prcpy(&self, other: &Other, src: &Other::Symbol, dst: &mut Self::Symbol)
        -> Result<(), Either<DevErr<Self>, DevErr<Other>>>
    {
        //! ### *default implementation*
        //! *`other`* --> *`main memory`* --> *`self`*
        self.load(Self::DatBox::from_vec(other.dump(src).map_err(Either::B)?.as_vec()), dst).map_err(Either::A)
    }
}