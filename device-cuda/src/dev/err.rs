pub(crate) use crate::raw::drv::cudaError_enum;

pub(crate) trait Wrap 
where Self: Sized {
    fn wrap<T>(self, v: T) -> Result<T, Self>;
}

impl Wrap for cudaError_enum {
    fn wrap<T>(self, v: T) -> Result<T, Self> {
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(v),
            e => Err(e)
        }
    }
}
