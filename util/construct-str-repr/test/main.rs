use trybuild::*;

#[test]
fn test_1() {
    let test = TestCases::new();
    test.pass("test/1-bare-struct.rs");
}

#[test]
fn test_2() {
    let test = TestCases::new();
    test.pass("test/2-bare-enum.rs");
}

#[test]
fn test_3() {
    let test = TestCases::new();
    test.pass("test/3-generic-struct.rs");
}

#[test]
fn test_4() {
    let test = TestCases::new();
    test.pass("test/4-generic-enum.rs");
}

#[test]
fn test_5() {
    let test = TestCases::new();
    test.pass("test/5-with-attrs.rs");
}