use crate::*;

/// common error format for devices
#[derive(Debug, Clone, Default)]
pub enum DevErr<D: Device> {
    /// print wanted and total
    MemNotEnough(String, D::DevErr),

    /// invalid access
    MemInvalidAccess(String, D::DevErr),

    /// function read from corrupted symbol (unintialized or written by failed functions)
    FuncReadCorrupted(String, D::DevErr),

    /// invalid input shape
    FuncInvalidInputShape(String, D::DevErr),

    /// invalid input type
    FuncInvalidInputType(String, D::DevErr),

    /// input on different device
    FuncInvalidInputDifferentDevice(String, D::DevErr),

    /// function not implemented
    FuncNotimplemented(String, D::DevErr),

    /// internal error caused a function to fail
    FuncInternalError(String, D::DevErr),

    /// device initialization failed with some reason
    InitFailure(String, D::DevErr),

    /// a fall back option
    #[default] Fallback,
}

impl<D: Device> DevErr<D> {
    pub fn prefix(self, x: String) -> Self {
        macro_rules! branch {
            ($($E: tt, )*) => {
                match self {
                    $($E(s, e) => $E(x+&s, e),)*
                    Fallback => Fallback, 
                }
            };
        }
        branch!(
            MemNotEnough, 
            MemInvalidAccess, 
            FuncReadCorrupted, 
            FuncInvalidInputShape,
            FuncInvalidInputType,
            FuncInvalidInputDifferentDevice,
            FuncNotimplemented,
            FuncInternalError,
            InitFailure,
        )
    }
}

pub use DevErr::*;