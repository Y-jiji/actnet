pub enum ExpHandle {
    External(PlaceHolder),
    Numerial(Numerial),
    BuiltIn(BuiltIn),
    Abs {
        binder: Binder,
        expression: Box<ExpHandle>,
    },
    App {
        function: Box<ExpHandle>,
        argument: Box<ExpHandle>,
    },
    Set {
        binder: Binder,
        expression: Box<ExpHandle>,
        r#where: Option<Box<ExpHandle>>,
    },
    Branch {
        r#where: Vec<(ExpHandle, ExpHandle)>,
        r#final: Box<ExpHandle>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<ExpHandle>,
        rhs: Box<ExpHandle>,
    },
    Unary {
        op: UnaryOp,
        rhs: Box<ExpHandle>,
    },
}

pub enum Binder {
    List(Vec<usize>),
    Just(usize),
}

pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    BitShl,
    BitShr,
    Gtr,
    Ltr,
    Gte,
    Lte,
}

pub enum UnaryOp {
    Neg,
    BitRvs,
}

pub enum Numerial {
    F32(f32),
    F64(f64),
    I32(i32),
    I16(i16),
    I8(i8),
    I64(i64),
    Bool(bool),
}

pub enum BuiltIn {
    Sum,
    Prd,
    Ord,
    Slv,
    Pow,
    Num(MathConst),
}

pub enum MathConst {
    E,  // 2.37
    Pi, // 3.14
}

pub struct PlaceHolder(usize);