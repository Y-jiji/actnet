use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::ops::{Index, IndexMut};

/// since every slot is of the same size, no fragmentation will happen (for the part of host memory)
/// however, there might be fragmentation issues for device memory
#[derive(Debug)]
pub(super)
struct SlotVec<T: Clone> {
    /// a slot vector
    v: Vec<(bool, T)>,
    /// a min-heap that retrieves the smallest empty slot
    q: BinaryHeap<Reverse<usize>>,
}

impl<T: Clone> SlotVec<T> {
    pub(super)
    fn new(v: Vec<T>) -> SlotVec<T> {SlotVec {
        v: v.into_iter().map(|x| {(true, x)}).collect(), 
        q: BinaryHeap::new(), 
    }}
    pub(super)
    fn put(&mut self, x: T) -> usize {
        match self.q.pop() {
            Some(Reverse(n)) => {self.v[n] = (true, x); n},
            None => {self.v.push((true, x)); self.v.len()-1},
        }
    }
    pub(super)
    fn unset(&mut self, n: usize) {
        self.q.push(Reverse(n));
        self.v[n].0 = false;
    }
}

impl<T: Clone> Index<usize> for SlotVec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        if !self.v[index].0 { panic!("this slot is marked as empty") }
        &self.v[index].1
    }
}

impl<T: Clone> IndexMut<usize> for SlotVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if !self.v[index].0 { panic!("this slot is marked as empty") }
        &mut self.v[index].1
    }
}