use ir_linajbr::*;

#[test]
fn test_1() {
    let x = compile! {
        <a, b> (x: [512, a; f32], y: [b, 786; f32]) -> [512, 786, a, b; f32] {
            return [i, j, k, l] -> x[i, k] * y[j, l];
        }
    };
    println!("{}", "@".repeat(100));
    println!("{x:?}");
    println!("{}", "@".repeat(100));
}