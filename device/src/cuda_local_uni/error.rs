use super::rawcuda::RawCudaError;

#[derive(Debug)]
pub(super)
enum DevErr {
    BoxNotFound,
    BoxNotEnough,
    Raw(RawCudaError),
    RecvError,
    ReplyError,
    NoReplySender,
}