use crate::*;

pub(crate) fn rand_unif_f<T>(upper: T, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> 
where rand::distributions::Standard: rand::distributions::Distribution<T>, 
      T: std::ops::Mul<Output=T> + Copy {
    for i in 0..len {
        unsafe{*a.ptr::<T>().add(i) = upper * rand::random::<T>()}
    }
    Ok(())
}

pub(crate) fn rand_unif_i<T>(upper: T, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> 
where rand::distributions::Standard: rand::distributions::Distribution<T>,
      T: std::ops::Rem<Output=T> + Copy {
    for i in 0..len {
        unsafe{*a.ptr::<T>().add(i) = rand::random::<T>() % upper}
    }
    Ok(())
}

pub(crate) fn rand_unif_b(upper: bool, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    if upper {unsafe{
        let len = bool::msize(len);
        for i in 0..(len/16) {*a.ptr::<u128>().add(i) = rand::random::<u128>()}
        for i in 0..(len%16) {*a.ptr::<u8>().add(i) = rand::random::<u8>()}
    }} else {unsafe{
        std::slice::
        from_raw_parts_mut(a.ptr::<u8>(), bool::msize(len)).fill(0);
    }}
    Ok(())
}

pub(crate) fn rand_unif(upper: WrapVal, len: usize, a: &mut ToySymbol) -> Result<(), DevErr<Toy>> {
    match upper {
        WrapVal::F32(upper) => rand_unif_f(upper, len, a),
        WrapVal::F64(upper) => rand_unif_f(upper, len, a),
        WrapVal::I32(upper) => rand_unif_i(upper, len, a),
        WrapVal::I64(upper) => rand_unif_i(upper, len, a),
        WrapVal::Bool(upper) => rand_unif_b(upper, len, a),
        _ => Ok(()),
    }
}