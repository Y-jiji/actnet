use std::{ops::{Index, IndexMut}, fmt::Debug};

/// a data structure that use index to substitute pointers
/// 
/// light-weight, usually faster than hash sets
#[derive(Debug, Clone)]
pub struct SlotVec<T: Clone> {
    /// a slot vector
    v: Vec<(bool, T)>,
    /// a stack to retrieve empty slot
    e: Vec<usize>,
}

impl<T: Clone + Debug> SlotVec<T> {
    #[inline]
    pub fn new(v: Vec<T>) -> SlotVec<T> {SlotVec {
        v: v.into_iter().map(|x| {(true, x)}).collect(), 
        e: Vec::new(), 
    }}
    #[inline]
    pub fn put(&mut self, x: T) -> usize {
        while let Some(n) = self.e.pop() {
            if n < self.v.len()
            { self.v[n] = (true, x); return n }
        }
        self.v.push((true, x)); self.v.len() - 1
    }
    #[inline]
    pub fn unset(&mut self, n: usize) {
        debug_assert!(
            n < self.v.len(), 
            "releasing a none-existing element"
        );
        debug_assert!(
            self.v[n].0,
            "releasing a none-existing element"
        );
        self.v[n].0 = false;
        self.e.push(n);
        while let Some(x) = self.v.pop() 
        { if x.0 { self.v.push(x); break; } }
    }
    #[inline]
    pub fn get(&self, i: usize) -> Option<&T> {
        if i > self.v.len() || !self.v[i].0 { None }
        else { Some(&self.v[i].1) }
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        if i > self.v.len() || !self.v[i].0 { None }
        else { Some(&mut self.v[i].1) }
    }
    #[inline]
    pub fn nofrag(&self) -> bool {
        for (m, _) in &self.v { if !m { return false } }
        return true
    }
}

impl<T: Clone + Debug> Index<usize> for SlotVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        self.get(i).expect("try to access an empty slot [{i:?}]")
    }
}

impl<T: Clone + Debug> IndexMut<usize> for SlotVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {   
        self.get_mut(i).expect("try to access an empty slot [{i:?}]")
    }
}

#[cfg(test)]
mod check_slot_vec {
    use rand::*;
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    #[test]
    /// there should be no fragmentation in a slot vector
    fn nofrag() {
        // initialize a slot vector
        let mut sv = SlotVec::new(Vec::<(i32, i32)>::new());
        let mut v = vec![(0i32, 0i32); 1<<15];
        let mut skip = vec![false; 1<<15];
        // randomly put some values
        for _ in 0..(1<<15) { let x = (random(), random()); v[sv.put(x)] = x; }
        assert!(sv.nofrag());
        let to_free : HashSet<_> = (0..(1<<12)).map(|_| {random::<usize>() % (1<<15)}).collect();
        // randomly unset some slots
        for n in to_free.iter() {
            sv.unset(*n);
            skip[*n] = true;
        }
        assert!(!sv.nofrag());
        // randomly put some values to fill fragments
        for _ in 0..to_free.len() { 
            let x = (random(), random());
            let n = sv.put(x);
            v[n] = x; skip[n] = false;
        }
        assert!(sv.nofrag());
        for n in 0..(1<<15) { assert!(skip[n] || v[n] == sv[n]) }
    }

    #[test]
    fn is_fast() {
        let v = Vec::from_iter((0..(1usize<<17)).map(|_| {random::<usize>() % (1<<15)}));
        let k = Vec::from_iter((0..(1usize<<17)).map(|_| {random::<usize>() % (1usize<<17)}));

        let start = Instant::now();
        let mut hm = HashMap::new();
        for n in 0..v.len() {hm.insert(n, v[n]);}
        for n in k.iter() {let x = hm.get(&n);}
        for n in 0..v.len() {hm.remove(&n);}
        let end = Instant::now();
        println!("{:?}", end-start);
        let hash_map_time = end-start;

        let start = Instant::now();
        let mut sv = SlotVec::new(vec![]);
        for n in 0..v.len() {sv.put(v[n]);}
        for n in k.iter() {let x = sv.get(*n);}
        for n in 0..v.len() {sv.unset(n);}
        let end = Instant::now();
        println!("{:?}", end-start);
        let slot_vec_time = end-start;

        // it should be at least 5 times faster than hash map version
        assert!(hash_map_time > 5 * slot_vec_time);
    }
}