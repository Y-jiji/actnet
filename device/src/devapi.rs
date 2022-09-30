pub trait DevAPI
where Self::ReplySend: ReplySender<Self>,
      Self: Sized,
{
    type DevErr;
    type DevBox;
    type ReplySend;
}

pub trait ReplySender<D: DevAPI> {
    fn send(msg: DevReply<D>) -> Result<(), D::DevErr>;
}

pub enum DevReply<D: DevAPI> {
    DevBox(D::DevBox),
    Void,
    DevErr(D::DevErr),
}

impl<D: DevAPI> DevReply<D> {
    pub(crate)
    fn to_devbox(self) -> Result<D::DevBox, D::DevErr> {
        match self {
            Self::DevBox(x) => Ok(x),
            Self::DevErr(x) => Err(x),
            _ => panic!("not devbox")
        }
    }
    pub(crate)
    fn to_void(self) -> Result<(), D::DevErr> {
        match self {
            Void => Ok(()),
            Self::DevErr(x) => Err(x),
            _ => panic!("not Void")
        }
    }
}