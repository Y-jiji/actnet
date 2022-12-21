/// data type
pub enum DType {I32, I64, U32, U64, F16, F32, F64}

/// scalar expression for primative types
pub enum MapFunc {
    Add(Box<MapFunc>, Box<MapFunc>),
    Sub(Box<MapFunc>, Box<MapFunc>),
    Mul(Box<MapFunc>, Box<MapFunc>),
    Div(Box<MapFunc>, Box<MapFunc>),
    Pow(Box<MapFunc>, Box<MapFunc>),
    Log(Box<MapFunc>, Box<MapFunc>),
    Max(Box<MapFunc>, Box<MapFunc>),
    Min(Box<MapFunc>, Box<MapFunc>),
    Gte(Box<MapFunc>, Box<MapFunc>),
    Gtr(Box<MapFunc>, Box<MapFunc>),
    Rem(Box<MapFunc>, Box<MapFunc>),
    Unif(DType, Vec<usize>),
    Norm(DType, Vec<usize>),
    Fill(DType, Vec<usize>, usize),
    Convert(DType, Box<MapFunc>),
    Axis(usize),
    Scalar(DType, usize),
    Access(usize, Vec<MapFunc>),
}

/// reduce expression
pub enum AggFunc {
    /// summation
    Sum(usize),
    /// cummulative summation
    CumSum(usize),
    /// max
    Max(usize),
    /// min
    Min(usize),
    /// argument sort
    ArgSort(usize),
}

/// variable with symbolic shape
pub enum NDExp {
    /// map with given expression arg0 on variables arg1, return a bundle of results
    Map(Vec<MapFunc>, Vec<NDExp>),
    /// aggregate with aggregation function arg0 on arg1
    Agg(AggFunc, Vec<NDExp>),
    /// change shape of arg1 to arg0
    Shape(Vec<usize>, Box<NDExp>),
    /// get an input
    Input(DType, Vec<usize>, usize),
}