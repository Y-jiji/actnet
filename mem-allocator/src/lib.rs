use slotvec::*;

use std::{ffi::c_void, ptr::null_mut};
use device_api::ComErr;

type Void = c_void;

#[derive(Debug, Clone)]
struct MemNode {
    /// free list next
    fl: usize,
    /// free list last
    fr: usize,
    /// physical list last
    pl: usize,
    /// physical list next
    pr: usize,
    /// size
    s: usize,
    /// pointer
    p: *mut Void
}

#[derive(Debug)]
pub struct MemShadow<const ALIGN: usize> {
    /// state of the memory, first nodes are free list heads
    state: SlotVec<MemNode>,
    /// the memory size it manages
    msize: usize,
    /// memory base
    mbase: *mut Void,
    /// level[i] = size upper bound for free list [i]
    level: Vec<usize>,
    /// usage accounting
    usage: usize,
}

impl<const ALIGN: usize> MemShadow<ALIGN> {
    /// create an empty MemShadow
    pub fn new(msize: usize, mbase: *mut Void, level: Vec<usize>) -> MemShadow<ALIGN> {
        let mut state = Vec::new();
        let ll = level.len();
        for i in 0..ll 
        { state.push(MemNode { fl: i, fr: i, pl: i, pr: i, s: 0, p: null_mut() }) }
        state.push(MemNode {fl: ll+1, fr: ll+1, pl: ll, pr: ll, s: 0, p: null_mut()});
        state.push(MemNode {fl: ll, fr: ll, pl: ll+1, pr: ll+1, s: msize, p: mbase});
        MemShadow { state: SlotVec::new(state), msize, mbase, level, usage: 0 }
    }
    /// input: size, return aligned size
    #[inline]
    fn align(s: usize) -> usize {
        debug_assert!(usize::MAX << ALIGN != 0, "1 << ALIGN >= usize::MAX");
        debug_assert!(ALIGN > 0);
        let mask = usize::MAX << ALIGN;
        (s + (1 << ALIGN) - 1) & mask
    }
    /// find a suitable level for a memory node
    #[inline]
    fn get_level(&self, s: usize) -> usize {
        for l in 0..self.level.len() {
            if s <= self.level[l]
            { return l }
        }
        self.level.len()
    }
    /// push node into free list
    #[inline]
    fn push_free(&mut self, n: usize) {
        let node = &self.state[n];
        let head = self.get_level(node.s & (usize::MAX ^ 1));
        let left = self.state[head].fl;
        self.state[left].fr = n;
        self.state[head].fl = n;
        self.state[n].fr = head;
        self.state[n].fl = left;
        debug_assert!(self.state[n].s & 1 == 1);
        self.state[n].s &= usize::MAX ^ 1;
    }
    /// remove a node from freelist and mark as not free
    #[inline]
    fn pull_free(&mut self, n: usize) {
        let node = &self.state[n];
        let l = node.fl;
        let r = node.fr;
        self.state[l].fr = r;
        self.state[r].fl = l;
        debug_assert!(self.state[n].s & 1 == 0);
        self.state[n].s |= 1;
    }
    /// find a suitable memory node, size greater than s
    #[inline]
    fn find(&self, s: usize) -> Option<usize> {
        let start = self.get_level(s.clone());
        let end = self.level.len() + 1;
        for head in start..end {
            let mut curs = self.state[head].fr;
            while curs != head {
                if self.state[curs].s >= s
                { return Some(curs) }
                curs = self.state[curs].fr;
            }
        }
        None
    }
    /// split a node and free part of them
    #[inline]
    fn split(&mut self, n: usize, s: usize) {
        if self.state[n].s < s + (1 << ALIGN) { return }
        debug_assert!(self.state[n].s & 1 == 1);
        let rh = MemNode {
            pl: n, pr: self.state[n].pr, 
            fl: 0, fr: 0, 
            s: self.state[n].s - s, 
            p: unsafe{self.state[n].p.add(s)} 
        };
        let rh = self.state.put(rh);
        let r = self.state[n].pr;
        self.state[r].pl = rh;
        self.state[n].pr = rh;
        self.push_free(rh);
        self.state[n].s = s | 1;
    }
    /// merge a node if the left or right is free
    /// 
    /// unset correspondent slot
    #[inline]
    fn merge(&mut self, mut n: usize) -> usize {
        debug_assert!(self.state[n].s & 1 == 1);
        let l = self.state[n].pl;
        let r = self.state[n].pr;
        let np = self.state[n].p;
        let ls = self.state[l].s;
        let lp = self.state[l].p;
        let rs = self.state[r].s;
        let rp = self.state[r].p;
        if rs & 1 == 0 && rp.gt(&np) {self.pull_free(r);}
        if ls & 1 == 0 && lp.lt(&np) {self.pull_free(l);}
        let mut merge_two = |a, b| {
            debug_assert!(self.state[a].pr == b);
            debug_assert!(self.state[b].pl == a);
            self.state[a].s += self.state[b].s & (usize::MAX ^ 1);
            let r = self.state[b].pr;
            self.state[a].pr = r;
            self.state[r].pl = a;
            self.state.unset(b);
        };
        if rs & 1 == 0 && rp.gt(&np) {merge_two(n, r);}
        if ls & 1 == 0 && lp.lt(&np) {merge_two(l, n); n = l;}
        return n;
    }
    /// find a suitable block and return
    pub fn alloc(&mut self, s: usize) -> Result<usize, ComErr> {
        let s = Self::align(s);
        let n = match self.find(s) { 
            None => Err({let (a, b) = self.usage(); ComErr::MemNotEnough(a + s, b)})?, 
            Some(n) => n 
        };
        self.pull_free(n);
        self.split(n, s);
        self.usage += self.state[n].s;
        debug_assert!(self.state[n].s & 1 == 1);
        Ok(n)
    }
    /// get pointer
    #[inline]
    pub fn getptr(&self, n: usize) -> *mut Void {
        self.state[n].p
    }
    /// get size
    #[inline]
    pub fn getsiz(&self, n: usize) -> usize {
        self.state[n].s
    }
    /// free a block
    pub fn free(&mut self, n: usize) -> Result<(), ComErr> {
        debug_assert!(self.state[n].s & 1 == 1);
        match self.state.get(n) {
            None => Err(ComErr::MemInvalidAccess)?,
            Some(x) => { self.usage += x.s }
        };
        let n = self.merge(n);
        self.push_free(n);
        Ok(())
    }
    /// get usage and total
    #[inline]
    pub fn usage(&self) -> (usize, usize) {
        (self.usage, self.msize)
    }
}

#[cfg(test)]
mod check_mem_shadow {
    use std::collections::{HashSet, HashMap};

    use super::*;
    use rand::{*, seq::IteratorRandom};

    fn print_free<const ALIGN: usize>(ms: &MemShadow<ALIGN>) {
        // number of levels
        let ls = ms.level.len();
        // final string output
        let mut st = String::new();
        for head in 0..(ls+1) {
            let hstr = format!("HEAD[{:?}, {:?}]", head, if head < ls { ms.level[head] } else { usize::MAX });
            st += "----------------\n";
            st += &hstr;
            let tab = vec![" "; hstr.len()].concat();
            let mut curs = ms.state[head].fr;
            let mut cnt = -1;
            while curs != head {
                if cnt > 0 && cnt % 4 == 3 { st += "\n"; st += &tab; }
                st += &format!(" -> {:?}", (ms.state[curs].p, ms.state[curs].s));
                curs = ms.state[curs].fr;
                cnt += 1;
            }
            st += "\n";
        }
        st += "----------------\n";
        st += "Physical";
        let tab = vec![" "; "Physical".len()].concat();
        let head = ms.state[ls].fr;
        let mut curs = head;
        let mut cnt = -1;
        loop {
            if cnt > 0 && cnt % 4 == 3 { st += "\n"; st += &tab; }
            st += &format!(" -> {:?}", ms.state[curs].p);
            curs = ms.state[curs].pr;
            cnt += 1;
            if curs == head { break; }
        }
        st += "\n";
        println!("{st}");
    }

    #[test]
    fn alloc() {
        let mut ms = MemShadow::<3>::new(128*1024*1024, null_mut(), vec![64, 128, 1024, 4096, 4096*1024]);
        print_free(&ms);
        let mut mh = Vec::new();
        for _ in 0..2048 {
            let s = random::<usize>() % (128 * 1023) + 1;
            println!("[[alloc {s}]]");
            print_free(&ms);
            let n = match ms.alloc(s) { Err(_)=>break, Ok(n)=>n };
            mh.push(n);
        }
        println!("{}", mh.len());
        for j in mh.iter() {
            println!("[[free {j}]]");
            ms.free(*j).unwrap();
        }
    }

    #[test]
    fn rolling() {
        let mut ms = MemShadow::<3>::new(128*1024*1024, null_mut(), vec![64, 128, 1024, 4096, 4096*1024]);
        let mut mh = HashMap::new();
        let mut usage = 0f32;
        let mut max_usage = 0f32;
        for _ in 0..2048 {
            let s = random::<usize>() % (64 * 1024) + 1;
            let n = match ms.alloc(s) { Err(_)=>break, Ok(n)=>n };
            mh.insert(n, s); usage += s as f32;
        }
        let mut delta = 0;
        loop {
            let mut rng = thread_rng();
            let mut is_free = HashSet::new();
            for n in mh.keys().choose_multiple(&mut rng, 256) {
                if is_free.contains(n) { continue; }
                is_free.insert(*n); ms.free(*n).unwrap();
            }
            for n in is_free.iter()
            { usage -= mh.remove(n).unwrap() as f32; }
            let mut cnt = 0;
            for _ in 0..256 {
                let s = random::<usize>() % (128 * 1023) + 1 + delta;
                let n = match ms.alloc(s) { Err(_)=>break, Ok(n)=>n };
                mh.insert(n, s);
                cnt += 1; usage += s as f32;
            }
            max_usage = f32::max(usage, max_usage);
            if cnt == 0 { break }
            delta += 1024;
        }
        let tot = ms.msize as f32;
        assert!(max_usage / tot > 0.9);
    }
}