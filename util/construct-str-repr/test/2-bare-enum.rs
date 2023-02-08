use construct_str_repr_pack::*;

#[derive(ConstructStrRepr)]
pub enum X {
    A {},
    B {a: String, b: usize},
    C(String, usize),
}

fn main() {
    let a = X::A{};
    let b = X::B{ a: "42".to_string(), b: 42usize };
    let c = X::C("42".to_string(), 42usize);
    eprintln!("{}", a.str_repr());
    eprintln!("{}", b.str_repr());
    eprintln!("{}", c.str_repr());
}