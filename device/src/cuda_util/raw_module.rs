use std::ptr::*;
use super::*;

#[derive(Debug)]
pub(crate) struct RawModule {
    pmodule: *mut CUmod_st,
}

const DEFAULT_OPS: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cuops.ptx"));

impl RawModule {
    fn new() -> Result<Self, RawCudaError> {
        Ok(Self {pmodule: Self::init_pmodule(&(DEFAULT_OPS.to_owned() + "\0"))?})
    }
    fn init_pmodule(image: &str) -> Result<*mut CUmod_st, RawCudaError> {
        let mut pmodule = null_mut::<CUmod_st>();
        match RawCudaError::from(unsafe{cuModuleLoadData(
                &mut pmodule as *mut _, 
                image as *const str as *const _)}
        ) {
            RawCudaError::CUDA_SUCCESS => {Ok(pmodule)},
            err => Err(err)
        }
    }
    fn get_func(&mut self, name: String) -> Result<*mut CUfunc_st, RawCudaError> {
        let mut func = null_mut::<CUfunc_st>();
        let pmodule = self.pmodule;
        let name = name.as_str().as_ptr() as *const i8;
        match RawCudaError::from(unsafe{cuModuleGetFunction(&mut func as *mut CUfunction, pmodule, name)}) {
            RawCudaError::CUDA_SUCCESS => Ok(func),
            err => Err(err),
        }
    }
}

impl Drop for RawModule {
    fn drop(&mut self) {
        match unsafe{cuModuleUnload(self.pmodule)} {
            RawCudaError::CUDA_SUCCESS => {}
            err => panic!("drop {:?} : {:?}", self, err)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn is_cuda_available() {
        // initialize cuda on current thread
        match RawCudaError::from(
            unsafe{cuInit(0)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("init cuda: {e:?}")}
        }
        // count device
        let mut device_count = 0;
        match RawCudaError::from(
            unsafe{cuDeviceGetCount(&mut device_count)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("get device count: {e:?}")}
        }
        assert!(device_count > 0, "device count: {device_count}");
        println!("avaible device: {device_count}");
    }

    #[test]
    fn raw_module_init() {
        match RawCudaError::from(
            unsafe{cuInit(0)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("cuda init {e:?}")}
        }
        let mut context = null_mut::<CUctx_st>();
        match RawCudaError::from(
            unsafe{cuCtxCreate_v2(&mut context, 0, 0)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("create context {e:?}")}
        }
        let raw_module = RawModule::new().unwrap();
        println!("{raw_module:?}");
        drop(raw_module);
    }

    #[test]
    fn raw_module_get_func() {
        match RawCudaError::from(
            unsafe{cuInit(0)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("cuda init {e:?}")}
        }
        let mut context = null_mut::<CUctx_st>();
        match RawCudaError::from(
            unsafe{cuCtxCreate_v2(&mut context, 0, 0)}
        ) {
            RawCudaError::CUDA_SUCCESS => {},
            e => {panic!("create context {e:?}")}
        }
        let mut raw_module = RawModule::new().unwrap();
        println!("{raw_module:?}");
        let func_ptr = loop {
            let func_ptr = match raw_module.get_func("add_f32".to_string()) {
                Ok(ptr) => ptr,
                Err(_) => continue,
            };
            println!("loop");
            break func_ptr
        };
        println!("{func_ptr:?}");
        drop(raw_module);
    }
}