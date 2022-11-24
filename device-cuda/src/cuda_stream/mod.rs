// a simple single stream executor (without pipelining)
mod stream;
pub use stream::*;

mod symbol;
pub use symbol::*;

mod datbox;
pub use datbox::*;

use crate::cuda_wrap::*;
use device_api::*;

#[cfg(test)]
mod check_cuda_stream {
    use super::*;

    #[test]
    fn new() {
        let cuda_stream = CudaStream::new(1024);
        drop(cuda_stream);
    }
}