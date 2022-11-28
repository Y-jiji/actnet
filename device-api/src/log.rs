//! this module defines basic logging operations for device

use crate::*;

/// if we want to bundle many streams into one stream, unexpected failure cannot be avoided. (e.g. power failure)
/// 
/// to make our system self-recovery, we have to implement checkpointing and rollback mechanisms for a computation stream
pub trait DevLog
where Self: Device {

    /// checkpoint type
    type Checkpoint;

    /// return a checkpoint of current device
    fn ckpt(&self) -> Result<Self::Checkpoint, (ComErr, Self::DevErr)>;

    /// roll back to the given checkpoint, it returns ok(...) only if checkpoint is generated by the latest successful call on self.ckpt()
    fn roll(&self, ckpt: &Self::Checkpoint) -> Result<(), (ComErr, Self::DevErr)>;
}