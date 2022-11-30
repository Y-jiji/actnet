# sctk-ndarray

## brief

This package implements an interface similar to numpy. 

## broadcasting

This package supports suffix-match broadcasting, which is weaker than numpy. 

To compromise this, we have powerfull general matrix multiplication and indexed operations. 

```rust
// a.shape: [6,3,5]
// b.shape: [3,5]
// this works because b's shape is a suffix of a's shape
(a + b).is_ok()
```

```rust
// a.shape == [6,1,5]
// b.shape == [3,5]
// this works because a's shape is not a suffix of b's shape and b's shape is not a suffix of a's shape
(a + b).is_err()
```