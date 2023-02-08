pub use repr::ConstructStrRepr;

// mod_path(ir_linajbr) is really dirty, since we want to pack it with a proc-macro lib
#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub enum Either<A, B>
where
    A: ConstructStrRepr, 
    B: ConstructStrRepr,
{ A(A), B(B) }

#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub struct LiteralBinder(pub String);

#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub struct NumerialBinder(pub usize);

#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub struct Arg<Bind: ConstructStrRepr>(pub Bind, pub Type<Bind>);

#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub struct Func<Bind> 
where Bind: ConstructStrRepr {
    pub args: Vec<Arg<Bind>>,
    pub body: Vec<Stmt<Bind>>,
    pub generic: Vec<Bind>,
    pub return_expr: Expr<Bind>,
    pub return_type: Type<Bind>,
}

#[non_exhaustive]
#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub enum Stmt<Bind>
where Bind: ConstructStrRepr {
    LetIn {
        bind: Bind,
        ty: Option<Type<Bind>>,
        expr: Expr<Bind>,
    },
}

#[non_exhaustive]
#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub enum Type<Bind> 
where Bind: ConstructStrRepr {
    F32,
    F64,
    I32,
    I64,
    Bool,
    // Array: (Shape(may contain generic args), ContentType)
    Arr(Vec<Either<usize, Bind>>, Box<Type<Bind>>),
    // Tuple: (Vec<Inner Types>)
    Tup(Vec<Type<Bind>>),
}

#[non_exhaustive]
#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub enum Expr<Bind>
where Bind: ConstructStrRepr {
    ArrBuild {
        bind: Vec<Bind>,
        expr: Box<Expr<Bind>>,
    },
    ArrIndex {
        arr: Box<Expr<Bind>>,
        idx: Vec<Expr<Bind>>,
    },
    SetBuild {
        expr: Box<Expr<Bind>>,
        bind: Vec<Bind>,
        cond: Box<Expr<Bind>>,
    },
    BinOps {
        ops: BinOps,
        lhs: Box<Expr<Bind>>,
        rhs: Box<Expr<Bind>>,
    },
    Tuple {
        elem: Vec<Expr<Bind>>
    },
    Call {
        func: Bind,
        args: Vec<Expr<Bind>>,
    },
    Bind(Bind),
}

#[non_exhaustive]
#[derive(Debug, ConstructStrRepr)]
#[mod_path(ir_linajbr)]
pub enum BinOps {
    Add,
    Mul,
    Sub,
    Div,
    Log,
    Pow,
}