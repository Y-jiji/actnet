use proc_macro::*;
use ir_linajbr_core::*;
use lazy_static::*;
use std::str::FromStr;

lazy_static!{
    static ref PARSER: FuncParser = FuncParser::new();
}

#[proc_macro]
pub fn compile(ts: TokenStream) -> TokenStream {
    let s = ts.to_string();
    // panic!("{s}");
    TokenStream::from_str(&PARSER.parse(&s).unwrap().str_repr()).unwrap()
}