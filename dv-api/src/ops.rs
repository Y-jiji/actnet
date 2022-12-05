//! this module defines basic operations and operands for a device and correspondent data box

use std::fmt::Debug;
use crate::*;


/* ------------------------------------------------------------------------------------------ */
/*                                       operation types                                      */
/* ------------------------------------------------------------------------------------------ */

#[derive(Debug)]
/// device function on device, will get even larger in near future
pub enum Func<'t, S: Debug + Symbol> {
/* ------------------------------- element-wise operations ------------------------------ */

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, b); o: (c, ); m: (len_a, len_b, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// len_a % len_b == 0 || len_b % len_a == 0
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..max(len_a, len_b) {
    ///     c[k] <-- a[k % len_a] + b[k % len_b]
    /// }
    /// ```
    Add {
        i: (&'t S, &'t S, ), 
        o: (&'t mut S, ),
        m: (usize, ),
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, b); o: (c, ); m: (len_a, len_b, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// len_a % len_b == 0 || len_b % len_a == 0
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..max(len_a, len_b) {
    ///     c[k] <-- a[k % len_a] - b[k % len_b]
    /// }
    /// ```
    Sub {
        i: (&'t S, &'t S, ), 
        o: (&'t mut S, ),
        m: (usize, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, b); o: (c, ); m: (len_a, len_b, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// len_a % len_b == 0 || len_b % len_a == 0
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..max(len_a, len_b) {
    ///     c[k] <-- a[k % len_a] * b[k % len_b]
    /// }
    /// ```
    Mul {
        i: (&'t S, &'t S, ), 
        o: (&'t mut S, ),
        m: (usize, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, b); o: (c, ); m: (len_a, len_b, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// len_a % len_b == 0 || len_b % len_a == 0
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..max(len_a, len_b) {
    ///     c[k] <-- a[k % len_a] / b[k % len_b]
    /// }
    /// ```
    Div {
        i: (&'t S, &'t S, ), 
        o: (&'t mut S, ),
        m: (usize, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, ); o: (b, ); m: (len, p, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// a and b are same-sized and same-typed
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..len_a {
    ///     b[k] <-- a[k] ** p
    /// }
    /// ```
    Pow {
        i: (&'t S, ),
        o: (&'t mut S, ),
        m: (usize, WrapVal, ),
    },
    
    /// @TODO(Y-jiji: exponential)
    Exp {
        i: (&'t S, ),
        o: (&'t mut S, ),
        m: (usize, WrapVal, ),
    },

    /// @TODO(Y-jiji: logrithm)
    Log {
        i: (&'t S, ),
        o: (&'t mut S, ),
        m: (usize, WrapVal, ),
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, ); o: (b, ); m: (len, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// a and b are same-sized and same-typed
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..len {
    ///     b[k] <-- -a[k]
    /// }
    /// ```
    Neg {
        i: (&'t S, ),
        o: (&'t mut S, ),
        m: (usize, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (); o: (a, ); m: (len, value, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// a's elements and value have the same type
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..len {
    ///     a[k] <-- value 
    /// }
    /// ```
    Fill {
        i: (),
        o: (&'t mut S, ),
        m: (usize, WrapVal, )
    },

// @TODO(Y-jiji: transpose, permute)

    /// ### *fields*
    /// ```pseudocode
    /// i: a
    /// o: b
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// b has same size as a
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// b <-- a
    /// ```
    Copy {
        /// source
        i: (&'t S, ), 
        /// destination
        o: (&'t mut S, ),
        /// none
        m: ()
    },

    /// transpose with given shape
    Transpose {
        /// source
        i: (&'t S, ),
        /// transposed symbol
        o: (&'t mut S, ),
        /// (shape, permutation)
        m: (&'t [usize], &'t [usize])
    },

/* ------------------------------------ random numbers ---------------------------------- */
    
    /// ### *fields*
    /// ```pseudocode
    /// i: (); o: (a, ); m: (len, upper, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// None
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..len {
    ///     sample a[k] uniformly from [0, upper)
    /// }
    /// ```
    RandUnif {
        i: (),
        o: (&'t mut S, ),
        m: (usize, WrapVal, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: (); o: (a, ); m: (len, )
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// None
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// for k in 0..len {
    ///     sample a[k] from standard normal distribution
    /// }
    /// ```
    RandSTDNorm {
        i: (),
        o: (&'t mut S, ),
        m: (usize, )
    },

/* ----------------------------------- compare and sort --------------------------------- */
// @TODO(Y-jiji: argument sort, argument max, argument min, top k)

    ArgSort {},

    ArgMax {},

    Max {},

    ArgMin {},

    Min {},

    TopK {},

    // less than
    LssThan {},

    // greater than
    GrtThan {},

/* ------------------------------------ gemm operations --------------------------------- */

    /// ### *fields*
    /// ```pseudocode
    /// i: (a, b, ); o: (c, ); m: (bat_a, bat_b, l0_a, l1_a, l2_a, l0_b, l1_b, l2_b)
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// l1_a == l1_b
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// let bat_c = max(bat_a, bat_b)
    /// 
    /// let l1_ab = l1_a
    /// 
    /// for i in 0..max(bat_a, bat_b) {
    ///     with a shape as (bat_a, l0_a, l1_a, l2_a)
    ///     with b shape as (bat_b, l0_b, l1_b, l2_b)
    ///     with c shape as (bat_c, l0_a, l0_b, l2_a, l2_b)
    ///     for (j0_a, j0_b, j1_ab, j2_a, j2_b) in 0..l0_a # 0..l0_b # 0..l1_ab # 0..l2_a # 0..l2_b {
    ///         c[i, j0_a, j2_a, j0_b, j2_b] <-- a[i%bat_a, j0_a, j1_ab, j2_a] * b[i%bat_b, j0_a, j1_ab, j2_b]
    ///     }
    /// }
    /// ```
    MMul {
        i: (&'t S, &'t S), 
        o: (&'t mut S, ),
        m: (usize, usize, usize, usize, usize, usize, usize, usize, )
    },

    /// ### *fields*
    /// ```pseudocode
    /// i: x
    /// o: y
    /// m: (i_sh, i_idx, o_idx)
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// i_sh[k] is shape of x[k]
    /// i_idx[k] have same length as i_sh 
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// @TODO(Y-jiji: write computational description for einstein summation convention)
    /// ```
    EinSum {
        i: &'t [&'t S],
        o: (&'t mut S, ),
        m: (&'t [&'t [usize]], &'t [&'t [usize]], &'t [usize])
    },

/* ----------------------------- linear algebraic operations -------------------------- */
//@TODO(Y-jiji: linear algebraic operations)

    /// pseudo inverse
    PseudoInv {},

    /// inverse
    Inv {},

    /// compute determinant
    Det {},

    /// singular value decomposition
    SingularValueD {},

    /// lower-upper decomposition
    LowerUpperD {},

    /// qr decomposition
    OrthTriangleD {},

    /// cholesky decomposition
    CholeskyD {},

/* --------------------------------- fall-back operation ------------------------------ */

    /// ### *fields*
    /// ```pseudocode
    /// None
    /// ```
    /// 
    /// ### *assumptions*
    /// ```pseudocode
    /// None
    /// ```
    /// 
    /// ### *effect*
    /// ```pseudocode
    /// you can write 
    /// match ... { 
    ///     ...
    ///     _ => Err(...)
    /// }
    /// without invocating warning, because this sentry value is left unmatched. 
    /// this is a recommended practice. branches are frequently added to this enum. 
    /// ```
    FallBack,
}