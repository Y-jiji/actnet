use device_api::*;

mod mem;

struct CudaStream;

type Void = std::ffi::c_void;

#[derive(Debug)]
struct DatBuf {
    p: *mut Void,
    s: usize,
    t: Type,
}

struct RawCudaError;

impl Device for CudaStream {
    type DatBuf = DatBuf;
    type DevBox = DatBuf;
    type DevErr = RawCudaError;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}