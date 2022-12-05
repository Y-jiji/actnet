use std::fmt::Debug;

use slotvec::*;

#[derive(Debug, Clone)]
struct TNode<T: Debug + Clone + Default> {
    /// successors
    succ: Vec<usize>,
    /// a packed value (free, done, unprepared predecessor count)
    pack: usize,
    /// additional information
    info: T,
}

#[derive(Debug, Clone)]
/// a data structure that maintains topology ordering in dag
pub struct TGraph<T: Debug + Clone + Default> {
    state : SlotVec<TNode<T>>,
    ready: Vec<(usize, T)>,
}

impl<T: Debug + Clone + Default> TGraph<T> {
    /// construct an empty graph
    pub fn new() -> TGraph<T> {TGraph {state: SlotVec::new(vec![]), ready: Vec::new()}}
    /// set waited predecessors, the output number will be return by todo function if this wait list is all done
    pub fn wait(&mut self, pred: &[usize], info: T) -> usize {
        if pred.len() > (usize::MAX>>2)
        { panic!("too many predecessors! is there really a function of {:?} arguments?", usize::MAX>>2) }
        let n = self.state.put(TNode{succ: Vec::new(), pack: pred.len(), info});
        let mut done = 0usize;
        for p in pred {
            let p = self.state.get_mut(*p).unwrap();
            if p.pack & !(usize::MAX >> 1) != 0 
            { panic!("referring to a piece of free data!"); }
            if p.pack & !(usize::MAX >> 2) != 0
            { done += 1 }
            p.succ.push(n);
        }
        self.state[n].pack -= done;
        if self.state[n].pack & (usize::MAX >> 2) == 0 {
            self.ready.push((n, std::mem::take(&mut self.state[n].info.clone())));
        }
        return n;
    }
    /// get ready functions (returned values should be taken care of by users)
    pub fn todo(&mut self) -> Vec<(usize, T)> {
        return std::mem::replace(&mut self.ready, Vec::new())
    }
    /// say some function is done
    pub fn done(&mut self, n: usize) {
        for s in self.state[n].succ.to_vec() {
            self.state[s].pack -= 1;
            if self.state[s].pack & (usize::MAX >> 2) == 0
            { self.ready.push((s, std::mem::take(&mut self.state[s].info))); }
        }
        self.state[n].pack |= 1 << (usize::BITS-2);
        if self.state[n].pack & (usize::MAX >> 2) != 0 {
            panic!("a function cannot be done before all arguments is done");
        }
        if self.state[n].pack >> (usize::BITS-2) == 0x3 {
            self.state.unset(n);
        }
    }
    /// set free bit to 0
    pub fn free(&mut self, n: usize) {
        self.state[n].pack |= 1 << (usize::BITS-1);
        if self.state[n].pack >> (usize::BITS-2) == 0x3 {
            self.state.unset(n);
        }
    }
}

#[cfg(test)]
mod check_tgraph {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use rand::{*, seq::IteratorRandom};

    #[derive(Debug, Clone)]
    struct StupidTNode<T: Debug+Clone+Default> {
        pred: Vec<usize>,
        done: bool,
        free: bool,
        info: T,
    }
    
    #[derive(Debug, Clone)]
    struct StupidTGraph<T: Debug+Clone+Default> {state: Vec<StupidTNode<T>>}

    impl<T: Debug+Clone+Default> StupidTGraph<T> {
        pub fn new() -> Self { StupidTGraph{state: Vec::new()} }
        pub fn wait(&mut self, pred: &[usize], info: T) -> usize {
            let n = self.state.len();
            self.state.push(StupidTNode { pred: pred.to_vec(), done: false, free: false, info });
            for p in pred { assert!(!self.state[*p].free); }
            return n;
        }
        pub fn done(&mut self, n: usize) { self.state[n].done = true; }
        pub fn free(&mut self, n: usize) { self.state[n].free = true; }
        pub fn todo(&mut self) -> Vec<(usize, T)> {
            // scan the whole list for 'emittable' functions
            self.state.iter().enumerate().filter_map(|(n, StupidTNode { pred, done, free, info })| { 
                if *done { return None; }
                assert!(!free);
                for p in pred { if !self.state[*p].done { return None; } }
                return Some((n, info.clone()));
            }).collect::<Vec<_>>()
        }
    }

    fn same_as_set(x: Vec<(usize, usize)>, y: Vec<(usize, usize)>) {
        let x = x.iter().map(|(_, n)| n).collect::<HashSet<_>>();
        let y = y.iter().map(|(_, n)| n).collect::<HashSet<_>>();
        assert!(x == y);
    }

    /// check if this tgraph have same behavior as the 'stupid' implementation
    /// it takes some time to drop the generated large structures
    #[test]
    fn check_tgraph() {
        // tensor graph
        let mut tg = TGraph::<usize>::new();
        let mut tg_inv = HashMap::new();
        // stupid graph
        let mut sg = StupidTGraph::<usize>::new();
        let mut sg_inv = HashMap::new();
        enum State { Free, Done, Todo }
        use State::*;
        let mut keep = HashMap::<usize, State>::new();
        for _ in 0..100000 {
            let tg_todo = tg.todo();
            let sg_todo = sg.todo();
            for (i, _) in tg_todo.iter() 
            { tg.done(*i); }
            for (i, _) in sg_todo.iter() 
            { sg.done(*i); }
            for (_, j) in sg_todo.iter() { 
                if random::<usize>() % 5 == 0 {
                    println!("done ({j})");
                    *keep.get_mut(j).unwrap() = Done;
                }
            }
            same_as_set(tg_todo, sg_todo);

            let mut will_free = Vec::new();

            for (k, v) in keep.iter() {
                match *v {
                    Done => if random::<usize>() % 15 == 0 {will_free.push(*k)},
                    // Todo => if random::<usize>() % 15 == 0 {will_free.push(*k)},
                    _ => (),
                }
            }

            for k in will_free {
                keep.insert(k, Free);
                println!("free ({k})");
                sg.free(sg_inv[&k]);
                tg.free(tg_inv[&k]);
            }

            let mut will_todo = Vec::new();

            for _ in 0..15 {
                let k = random::<usize>() % 1000usize;
                if keep.get(&k).is_none() {
                    will_todo.push(k);
                }
            }

            for k in will_todo {
                let mut rng = thread_rng();
                let pred = keep.iter().choose_multiple(&mut rng, 5).iter()
                    .filter_map(|(k, v)| match v { Free => None , _ => Some(**k) }).collect::<Vec<_>>();
                println!("wait ({k} -> {pred:?})");
                let sg_pred = pred.iter().map(|p| { sg_inv[p] }).collect::<Vec<_>>();
                sg_inv.insert(k, sg.wait(&sg_pred, k));
                let tg_pred = pred.iter().map(|p| { tg_inv[p] }).collect::<Vec<_>>();
                let tg_id = tg.wait(&tg_pred, k);
                tg_inv.insert(k, tg_id);
                keep.insert(k, Todo);
            }
        }
    }
}