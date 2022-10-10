pub type Void = std::ffi::c_void;

/// modified stream computation model API
pub trait Device {
    /// memory box on host
    type HBox;
    /// memory box on device
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
    /// ops: operation name string
    fn launch<I, F>(
        &mut self, ops: String,
        src: [Option<&Self::DBox>; 6], dst: &mut Self::DBox,
        meta_i: [I; 6] , meta_f: [F; 6]
    ) -> Result<(), Self::DErr> 
    { todo!("launch"); }

    /// add a callback
    fn add_hook(&mut self, callback: Box<dyn FnOnce() + Send>) -> Result<(), Self::DErr>
    { todo!("add_hook"); }

    /// inspect the content inside a box
    fn inspect(&mut self, src: &Self::DBox) -> String
    { todo!("inspect"); }
}

pub mod cuda_util;
pub mod mem_util;

mod cuda_uni_device;
pub use cuda_uni_device::*;

mod cuda_uni_stream;
pub use cuda_uni_stream::*;