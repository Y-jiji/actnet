pub enum Either<A, B> { A(A), B(B) }

pub struct Func {
    pub args: Vec<(String, Type)>,
    pub body: Vec<Stmt>,
    pub return_expr: Expr,
    pub generic: Vec<String>,
    pub return_type: Type,
}

#[non_exhaustive]
pub enum Stmt {
    LetIn {
        bind: String,
        ty: Option<Type>,
        expr: Expr,
    },
}

#[non_exhaustive]
pub enum Type {
    F32,
    F64,
    I32,
    I64,
    Bool,
    // Array: (Shape(may contain generic args), ContentType)
    Arr(Vec<Either<usize, String>>, Box<Type>),
}

#[non_exhaustive]
pub enum Expr {
    ArrBuild {
        bind: Vec<String>,
        expr: Box<Expr>,
    },
    ArrIndex {
        arr: Box<Expr>,
        idx: Vec<Expr>,
    },
    SetBuild {
        expr: Box<Expr>,
        bind: Vec<String>,
        cond: Box<Expr>,
    },
    BinOps {
        ops: BinOps,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Call {
        func: String,
        args: Vec<Expr>,
    },
    Bind(String),
}

#[non_exhaustive]
pub enum BinOps {
    Add,
    Mul,
    Sub,
    Div,
    Log,
    Pow,
}