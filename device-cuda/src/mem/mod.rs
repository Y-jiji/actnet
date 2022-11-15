mod slot_vec;
use slot_vec::*;
use std::{ffi::c_void, marker::PhantomData};

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
pub(crate)
struct MemShadow<const ALIGN: usize> {
    /// state of the memory, first nodes are free list heads
    state: SlotVec<MemNode>,
    /// the memory size it manages
    msize: usize,
    /// memory base
    mbase: *mut Void,
    /// level[i] = size upper bound for free list [i]
    level: [usize],
}

impl<const ALIGN: usize> MemShadow<ALIGN> {
    /// align funtion
    #[inline]
    fn align(s: usize) -> usize {
        #[cfg(debug_assertions)]
        assert!(usize::MAX << ALIGN != 0, "1 << ALIGN >= usize::MAX");
        (s + usize::MAX << ALIGN - 1) & (usize::MAX << ALIGN)
    }
    /// find a suitabel level for a memory node
    fn level_for(&self, s: usize) -> usize {
        let mut l = 0; 
        loop {
            if l == self.level.len() - 1 ||
               s <= self.level[l]
            { break l }
            l += 1;
        }
    }
    /// find a suitable memory node, size greater than s
    fn find(&self, s: usize) -> Option<usize> {
        let s = Self::align(s);
        // the free list head to start from
        let l = self.level_for(s);
        // find a free block in free lists
        for head in l..(self.level.len()+1) {
            let mut curs = self.state[head].fr;
            while curs != head {
                if self.state[curs].s >= s 
                { return Some(curs) }
                curs = self.state[curs].fr;
            }
        }
        return None;
    }
    /// remove a node from freelist and mark as not free
    fn unfree(&mut self, n: usize) {
        let l = self.state[n].fl;
        let r = self.state[n].fr;
        self.state[n].s |= 1;
        self.state[l].fr = r;
        self.state[r].fl = l;
    }
    /// split a node and free part of them
    fn split(&mut self, n: usize, s: usize) {
        debug_assert!(s == Self::align(s));
        if self.state[n].s < 8 + s { return }
        let ms = self.state[n].s - s;
        self.state[n].s = s;
        let m = self.state.put(MemNode { 
            fl: 0, fr: 0, pl: 0, pr: 0, s: ms, 
            p: unsafe{self.state[n].p.add(s)} });
        self.link_free(m);
    }
    /// merge a node if the left or right is free
    fn merge(&mut self, mut n: usize) -> usize {
        let l  = self.state[n].pl;
        let ls = self.state[l].s;
        debug_assert!(ls == Self::align(ls));
        let r  = self.state[n].pr;
        let rs = self.state[r].s;
        debug_assert!(rs == Self::align(rs));
        let mut m_two = |n, m| {
            // m: the block to merge
            // n: the block to merge to
            self.state[n].s += self.state[m].s;
            self.unfree(m);
            let mr = self.state[m].pr;
            self.state[n].pr = mr;
            self.state[mr].pl = n;
            self.state.unset(m);
        };
        if rs & 1 == 0 {
            let m = r;
            m_two(n, m);
        }
        if ls & 1 == 0 {
            let m = n; n = l;
            m_two(n, m);
        }
        return n;
    }
    fn link_free(&mut self, n: usize) {
        // free a memory node
        self.state[n].s &= usize::MAX ^ 1;
        // find the size
        let s = self.state[n].s;
        // put back to free list
        let h = self.level_for(s); // first find correct level
        let l = self.state[h].fl;  // find place to insert after
        self.state[n].fr = h;
        self.state[n].fl = l;
        self.state[h].fl = n;
        self.state[l].fr = n;
    }
    /// find a suitable block and return
    pub(crate)
    fn alloc(&mut self, s: usize) -> Option<usize> {
        let s = Self::align(s);
        if let Some(n) = self.find(s) {
            self.split(n, s);
            self.unfree(n);
            return Some(n);
        }
        return None;
    }
    /// free a block
    pub(crate)
    fn free(&mut self, n: usize) {
        let n = self.merge(n);
        self.link_free(n);
    }
}