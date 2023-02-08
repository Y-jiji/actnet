use construct_str_repr_pack::*;

#[derive(ConstructStrRepr)]
pub struct A {
    pub a: String,
    pub b: usize,
}

#[derive(ConstructStrRepr)]
pub struct B(String, usize);

fn main() {
    let a = A { a: "42".to_string(), b: 42usize };
    eprintln!("{}", a.str_repr());
    let b = B("42".to_string(), 42usize);
    eprintln!("{}", b.str_repr());
}