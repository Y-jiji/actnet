use dvapi::*;

struct Tensor<'a, D: Device> {
    /// [v]alue
    v: Option<D::Symbol>,
    /// [g]radient
    g: Option<D::Symbol>,
    /// computed from these [p]redecessors
    p: &'a [Tensor<'a, D>],
    /// computed with this [f]unction
    f: Func<'a, D::Symbol>,
    /// device
    d: &'a D,
}

#[cfg(test)]
mod check_tensor {
}