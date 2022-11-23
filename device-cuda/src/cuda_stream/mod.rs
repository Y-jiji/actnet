// a simple single stream executor (without pipelining)
mod stream;
pub use stream::*;

mod devbox;
pub use devbox::*;

mod datbuf;
pub use datbuf::*;

use crate::cuda_wrap::*;
use device_api::*;

impl Device for Stream {
    type DatBuf = DatBuf;
    type DevBox = DevBox;
    type DevErr = cudaError_enum;

    fn delbox(&self, devbox: &mut Self::DevBox) -> Result<(), (ComErr, Self::DevErr)> {
    }
}