use crate::*;

pub(crate) fn copy(a: &ToySymbol, b: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    unsafe{copy_nonoverlapping(a.inner, b.inner, a.msize)};
    Ok(())
}
