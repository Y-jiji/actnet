use construct_str_repr_pack::*;

#[derive(ConstructStrRepr)]
pub enum A<X: ConstructStrRepr, Y>
where Y: Clone + ConstructStrRepr {
    X(X),
    Y(Y),
}

fn main() {
    let x = A::<String, usize>::X("42".to_owned());
    let y = A::<String, usize>::Y(42);
    eprintln!("{}", x.str_repr());
    eprintln!("{}", y.str_repr());
}