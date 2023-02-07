#[derive(Debug)]
pub enum Either<A, B> { A(A), B(B) }

#[derive(Debug)]
pub struct LiteralBinder(pub String);

#[derive(Debug)]
pub struct Func<Binder=LiteralBinder> {
    pub args: Vec<(Binder, Type<Binder>)>,
    pub body: Vec<Stmt<Binder>>,
    pub generic: Vec<Binder>,
    pub return_expr: Expr<Binder>,
    pub return_type: Type<Binder>,
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Stmt<Binder=LiteralBinder> {
    LetIn {
        bind: Binder,
        ty: Option<Type<Binder>>,
        expr: Expr<Binder>,
    },
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Type<Binder=LiteralBinder> {
    F32,
    F64,
    I32,
    I64,
    Bool,
    // Array: (Shape(may contain generic args), ContentType)
    Arr(Vec<Either<usize, Binder>>, Box<Type<Binder>>),
}

#[non_exhaustive]
#[derive(Debug)]
pub enum Expr<Binder=LiteralBinder> {
    ArrBuild {
        bind: Vec<Binder>,
        expr: Box<Expr<Binder>>,
    },
    ArrIndex {
        arr: Box<Expr<Binder>>,
        idx: Vec<Expr<Binder>>,
    },
    SetBuild {
        expr: Box<Expr<Binder>>,
        bind: Vec<Binder>,
        cond: Box<Expr<Binder>>,
    },
    BinOps {
        ops: BinOps,
        lhs: Box<Expr<Binder>>,
        rhs: Box<Expr<Binder>>,
    },
    Call {
        func: Binder,
        args: Vec<Expr<Binder>>,
    },
    Bind(Binder),
}

#[non_exhaustive]
#[derive(Debug)]
pub enum BinOps {
    Add,
    Mul,
    Sub,
    Div,
    Log,
    Pow,
}