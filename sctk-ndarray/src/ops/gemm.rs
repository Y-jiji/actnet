//! general matrix multiplication
use crate::*;

//@TODO(Y-jiji: implement gemm)
trait GeMM<D: Device>
where Self: Sized {
    /* ------------------------------- abbrievated ------------------------------- */
    /// alias for matrix multiplication
    fn mm(&self, rhs: &Self) -> Result<Self, DevErr<D>>;
    /// alias for dot_product
    fn dot(&self, rhs: &Self) -> Result<Self, DevErr<D>>;
    /// alias for einstein summation convention
    fn ein(o: &[usize], idx: &[&[usize]], x: &[&Self]) -> Result<Self, DevErr<D>>;
    /* -------------------------------- full name -------------------------------- */
    /// \sum_j a[i][j] * b[j][k] = c[i][k]
    fn matrix_multplication(&self, rhs: &Self) -> Result<Self, DevErr<D>>;
    /// \sum_j a[i][j] * b[k][j] = c[i][k]
    fn dot_product(&self, rhs: &Self) -> Result<Self, DevErr<D>>;
    /// einstein summation convention, index is (input index, output index)
    /// @TODO(Y-jiji: implement proc_macro for einsum)
    fn einstein(out_index: &[usize], operand_index: &[&[usize]], operand: &[&Self]) -> Result<Self, DevErr<D>>;
    /// mean for one dimension
    fn mean(&self, at: usize) -> Result<Self, DevErr<D>>;
    /// sum for one dimension
    fn sum(&self, at: usize) -> Result<Self, DevErr<D>>;
}

//@TODO(Y-jiji: implement reindex operations)
trait ReIndex<D: Device>
where Self: Sized {
    /* ------------------------------- abbrievated ------------------------------- */
    /// alias for reshape
    fn r(&self, shape: &[usize]) -> Result<Self, DevErr<D>>;
    /// alias for transpose
    fn t(&self, index: &[usize]) -> Result<Self, DevErr<D>>;
    /// alias for duplicate
    fn d(&self, shape: &[usize]) -> Result<Self, DevErr<D>>;
    /* -------------------------------- full name -------------------------------- */
    /// reshape, zero stands for flattening and should only appear once in shape
    fn reshape(&self, shape: &[usize]) -> Result<Self, DevErr<D>>;
    /// tranpose, index should be a permutation
    fn transpose(&self, index: &[usize]) -> Result<Self, DevErr<D>>;
    /// duplicate with given shape, e.g. a[i][j][k]
    fn duplicate(&self, shape: &[usize]) -> Result<Self, DevErr<D>>;
}