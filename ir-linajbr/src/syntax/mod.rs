mod syntax_rule;
mod syntax;
pub use syntax_rule::*;
pub use syntax::*;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn see_parse_tree() {
        let parser = FuncParser::new();
        let parsed = parser.parse("
            <a, b> (x:[512, a; f32], y:[b, 786; f32] ) 
            -> [512, 786, a, b; f32] {
                return [i, j, k, l] -> x[i, k] * y[j, l];
            }
        ").unwrap();
        println!("{parsed:?}");
    }
}