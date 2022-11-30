use std::{fmt::Debug, collections::HashSet};

use slotvec::*;
use device_api::*;

#[derive(Debug, Clone)]
struct TNode<T: Debug + Clone> {
    /// predcessors
    pred: Vec<usize>,
    /// successors
    succ: Vec<usize>,
    /// additional information
    info: T,
}

#[derive(Debug, Clone)]
pub struct TGraph<T: Debug + Clone> {
    state : SlotVec<TNode<T>>,
    ready : Vec<Vec<usize>>,
}

impl<T: Debug + Clone> TGraph<T> {
    pub fn new() -> TGraph<T> {TGraph {
        state: SlotVec::new(vec![]),
        ready: Vec::new(),
    }}
}