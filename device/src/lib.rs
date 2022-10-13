pub type Void = std::ffi::c_void;

/// modified stream computation model API
/// 
/// : in stream computation model, reading values associated to boxes is allowed only on stream termination
/// 
/// : this modified version returns value on memory box deletion, and add inspect for debugging
pub trait Device {
    /// memory box on host
    type HBox;
    /// memory box on device, should be ?Drop, ?Copy and ?Clone
    type DBox;
    /// device error
    type DErr;

    /// allocate a new box on device, initialize data from host
    fn new_box(&mut self, src: Self::HBox) -> Result<Self::DBox, Self::DErr> 
    { todo!("new_box"); } // <--------- avoid some annoying compilation error when implementing this trait for other structs

    /// delete and copy data back
    fn del_box(&mut self, src: Self::DBox) -> Result<Self::HBox, Self::DErr> 
    { todo!("del_box"); }

    /// copy box data from source to destination (a special operation)
    fn cpy_box(&mut self, src: &Self::DBox, dst: &mut Self::DBox) -> Result<(), Self::DErr> 
    { todo!("cpy_box"); }

    /// add an operation launch to device
    fn launch<I, F>(
        // ops: operation name string
        &mut self, ops: String,
        // dst = ops(src[0], src[1], ..., src[5], meta_i, meta_f)
        src: [Option<&Self::DBox>; 6], dst: &mut Self::DBox,
        // interger parameters, float parameters
        _int: [I; 6] , _flt: [F; 2]
    ) -> Result<(), Self::DErr> 
    { todo!("launch"); }

    /// add a callback
    fn add_hook(&mut self, callback: Box<dyn FnOnce() + Send>) -> Result<(), Self::DErr>
    { todo!("add_hook"); }

    /// inspect the content inside a box
    fn inspect(&mut self, src: &Self::DBox) -> String
    { todo!("inspect"); }
}


pub trait BridgeFrom<SRC: Device> where Self: Device {
    fn cpy_box(&mut self, src: &mut SRC::DBox, dst: &mut Self::DBox) -> Result<(), Self::DErr>
    { todo!("cpy_box"); }
    fn mov_box(&mut self, src: SRC::DBox) -> Result<Self::DBox, Self::DErr>
    { todo!("mov_box"); }
}

pub mod cuda_util;
pub mod mem_util;

mod cuda_uni_device;
pub use cuda_uni_device::*;

mod cuda_uni_stream;
pub use cuda_uni_stream::*;

