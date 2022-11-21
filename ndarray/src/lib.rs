use device_api::*;

mod display;
use display::*;

mod ops;
use ops::*;

struct NDArray<D: Device> {
    devbox: D::DevBox,
    device: D,
    shape: Vec<usize>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
