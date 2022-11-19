use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::ops::{Index, IndexMut};

/// since every slot is of the same size, no fragmentation will happen (for the part of host memory)
/// however, there might be fragmentation issues for device memory
#[derive(Debug)]
pub struct SlotVec<T: Clone> {
    /// a slot vector
    v: Vec<(bool, T)>,
    /// a min-heap that retrieves the smallest empty slot
    q: BinaryHeap<Reverse<usize>>,
}

impl<T: Clone> SlotVec<T> {
    pub fn new(v: Vec<T>) -> SlotVec<T> {SlotVec {
        v: v.into_iter().map(|x| {(true, x)}).collect(), 
        q: BinaryHeap::new(), 
    }}
    pub fn put(&mut self, x: T) -> usize {
        match self.q.pop() {
            Some(Reverse(n)) => {self.v[n] = (true, x); n},
            None => {self.v.push((true, x)); self.v.len()-1},
        }
    }
    pub fn unset(&mut self, n: usize) {
        if !self.v[n].0 {return}
        self.q.push(Reverse(n));
        self.v[n].0 = false;
    }
    pub fn nofrag(&self) -> bool {
        for (m, _) in &self.v { if !m { return false } }
        return true
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

#[cfg(test)]
mod check_slot_vec {
    use rand::*;
    use super::*;

    #[test]
    /// there should be no fragmentation in a slot vector
    fn nofrag() {
        // initialize a slot vector
        let mut sv = SlotVec::new(Vec::<(i32, i32)>::new());
        let mut v = Vec::<(bool, (i32, i32))>::new();
        // randomly put some values
        for _ in 0..1024 { let x = (random(), random()); sv.put(x); v.push((true, x)); }
        assert!(sv.nofrag());
        assert!(sv.v == v);
        // randomly unset some slots
        for _ in 0..128 { sv.unset(random::<usize>() % 1024); }
        assert!(!sv.nofrag());
        // randomly put some values to fill fragments
        for _ in 0..128 { sv.put((random(), random())); }
        assert!(sv.nofrag());
    }
}