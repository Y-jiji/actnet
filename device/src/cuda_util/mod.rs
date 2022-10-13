mod raw_stream;
mod raw_module;
mod cuda;

use cuda::*;
use raw_stream::*;
use raw_module::*;


#[cfg(test)]
mod test {
    use std::mem::size_of;
    use std::ptr::*;

    use super::*;
    use crate::*;

    /// This is f**king messy, wrapped up version in cuda_uni_device and cuda_uni_stream
    #[test]
    fn simple_add() {
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
        let mut raw_stream = RawStream::new().unwrap();
        println!("raw stream: {raw_stream:?}");
        let mut raw_module = RawModule::new().unwrap();
        let add_handle = raw_module.get_func("add_f32".to_string()).unwrap();
        println!("function handle: {add_handle:?}");
        let len_byte = 128 * size_of::<f32>();
        let ptr_base = raw_stream.malloc(len_byte * 3).unwrap();
        
        // compute arr_z[i] = arr_x[i] + arr_y[i] for every i
        let arr_x = vec![17f32; 128];
        let ptr_x = ptr_base as *mut _;
        raw_stream.memcpy(
            DPtr::Host(arr_x.as_ptr() as *mut _), 
            DPtr::Device(ptr_base as *mut _), len_byte
        ).unwrap();
        println!("copy arr_x");
        let arr_y = vec![15f32; 128];
        let ptr_y = unsafe{ptr_base.add(len_byte)} as *mut _;
        raw_stream.memcpy(
            DPtr::Host(arr_y.as_ptr() as *mut _), 
            DPtr::Device(ptr_y), len_byte
        ).unwrap();
        println!("copy arr_y");
        let mut arr_z = vec![00f32; 128];
        let ptr_z = unsafe{ptr_base.add(2*len_byte)} as *mut _;
        // launch a function on target stream
        raw_stream.launch(
            add_handle, ((1,1,1), (512, 1, 1), 0), 
            &[
                &ptr_x  as *const _ as *mut Void,
                &ptr_y  as *const _ as *mut Void,
                &ptr_z  as *const _ as *mut Void,
                &128u64 as *const _ as *mut Void,
            ]
        ).unwrap();
        println!("function launch");

        // copy back after the previous job finished
        raw_stream.memcpy(
            DPtr::Device(ptr_z), 
            DPtr::Host(arr_z.as_ptr() as *mut _), 
            len_byte
        ).unwrap();

        println!("copy back to arr_z");
        drop(raw_stream);
        drop(raw_module);
        assert!(arr_z == vec![32f32; 128]);
    }
}