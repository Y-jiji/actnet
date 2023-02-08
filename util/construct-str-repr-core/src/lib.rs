pub trait ConstructStrRepr {
    fn str_repr(self) -> String;
}

impl<T> ConstructStrRepr for Option<T>
where
    T: ConstructStrRepr
{
    fn str_repr(self) -> String {
        match self {
            Some(x) => format!("std::option::Option::Some({})", x.str_repr()),
            None => format!("std::option::Option::None")
        }
    }
}

impl<T> ConstructStrRepr for Box<T>
where
    T: ConstructStrRepr,
{
    fn str_repr(self) -> String {
        format!("std::boxed::Box::new({})", (*self).str_repr())
    }
}

impl<T> ConstructStrRepr for Vec<T>
where
    T: ConstructStrRepr,
{
    fn str_repr(self) -> String {
        let a = self
            .into_iter()
            .fold(String::new(), |x, y| x + &(y.str_repr()) + ",");
        format!("std::vec![{}]", a)
    }
}

impl ConstructStrRepr for String {
    fn str_repr(self) -> String {
        format!("\"{self}\".to_string()")
    }
}

impl ConstructStrRepr for &str {
    fn str_repr(self) -> String {
        format!("\"{self}\"")
    }
}

macro_rules! impl_primary {
    ($x: ident) => {
        impl ConstructStrRepr for $x {
            fn str_repr(self) -> String {
                format!("({self} as {})", stringify!($x))
            }
        }
    };
}

impl_primary!(usize);
impl_primary!(isize);
impl_primary!(u32);
impl_primary!(i32);
impl_primary!(u64);
impl_primary!(i64);
impl_primary!(u128);
impl_primary!(i128);
impl_primary!(u16);
impl_primary!(i16);
impl_primary!(u8);
impl_primary!(i8);
impl_primary!(f32);
impl_primary!(f64);