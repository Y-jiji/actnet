//! general matrix multiplication
use crate::*;

//@TODO implement gemm
trait GeMM<D: Device>
where Self: Sized {
    /* ------------------------------- abbrievated ------------------------------- */
    /// alias for matrix multiplication
    fn mm(&self, rhs: &Self) -> Result<Self, TupErr<D>>;
    /// alias for dot_product
    fn dot(&self, rhs: &Self) -> Result<Self, TupErr<D>>;
    /// alias for einstein
    fn e(outshape: &[usize], index: &[&[usize]], input: &[&Self]) -> Result<Self, TupErr<D>>;
    /* -------------------------------- full name -------------------------------- */
    /// \sum_j a[i][j] * b[j][k] = c[i][k]
    fn matrix_multplication(&self, rhs: &Self) -> Result<Self, TupErr<D>>;
    /// \sum_j a[i][j] * b[k][j] = c[i][k]
    fn dot_product(&self, rhs: &Self) -> Result<Self, TupErr<D>>;
    /// einstein summation convention, index is (input index, output index)
    fn einstein(outshape: &[usize], index: &[&[usize]], input: &[&Self]) -> Result<Self, TupErr<D>>;
    /// mean for one dimension
    fn mean(&self, at: usize) -> Result<Self, TupErr<D>>;
    /// sum for one dimension
    fn sum(&self, at: usize) -> Result<Self, TupErr<D>>;
}

//@TODO implement reindex operations
trait ReIndex<D: Device>
where Self: Sized {
    /* ------------------------------- abbrievated ------------------------------- */
    /// alias for reshape
    fn r(&self, shape: &[usize]) -> Result<Self, TupErr<D>>;
    /// alias for transpose
    fn t(&self, index: &[usize]) -> Result<Self, TupErr<D>>;
    /// alias for duplicate
    fn d(&self, shape: &[usize]) -> Result<Self, TupErr<D>>;
    /* -------------------------------- full name -------------------------------- */
    /// reshape, zero stands for flattening and should only appear once in shape
    fn reshape(&self, shape: &[usize]) -> Result<Self, TupErr<D>>;
    /// tranpose, index should be a permutation
    fn transpose(&self, index: &[usize]) -> Result<Self, TupErr<D>>;
    /// duplicate with given shape, e.g. a[i][j][k]
    fn duplicate(&self, shape: &[usize]) -> Result<Self, TupErr<D>>;
}