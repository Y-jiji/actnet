use construct_str_repr_pack::*;

#[derive(ConstructStrRepr)]
pub struct A<X: ConstructStrRepr, Y>
where Y: Clone + ConstructStrRepr {
    x: X,
    y: Y,
}

fn main() {
    let a = A { x: "42".to_string(), y: 42 } ;
    eprintln!("{}", a.str_repr());
}