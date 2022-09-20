use std::{collections::{HashMap, BTreeSet}, marker::PhantomPinned};
use super::{Dev, Void};
use super::error::DevErr;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super)
struct DevBox {
    /// actual pointer to memory
    ptr: *mut Void,
    /// device type
    dev: Dev,
    /// next device box
    nxt: *mut DevBox,
    /// last device box
    lst: *mut DevBox,
    /// box id for back reference
    bid: usize,
    /// box size
    len: usize,
    /// a device box refers to next/last device box, thus should be pinned in memory
    pin: PhantomPinned,
}

#[derive(Debug)]
pub(super)
struct ShadowMem {
    /// memory in use
    mem_used: [HashMap<usize, *mut DevBox>; 16],
    /// free gpu memory
    gpu_free: [BTreeSet<(usize, *mut DevBox)>; 16],
    /// lower pointer for gpu memory
    glower: *mut Void,
    /// upper pointer for gpu memory
    gupper: *mut Void,
    /// free cpu memory
    cpu_free: [BTreeSet<(usize, *mut DevBox)>; 16],
    /// lower pointer for cpu memory
    clower: *mut Void,
    /// upper pointer for cpu memory
    cupper: *mut Void,
}

unsafe impl Send for ShadowMem {}

impl ShadowMem {
    /// coleacse with next and last box if possible
    fn merge(&mut self, devbox: &mut DevBox) -> Result<(), DevErr> {
        let shard_id = devbox.bid & 0xffusize;
        let next = unsafe{&mut *devbox.nxt};
        // try coleacse with next and last box
        if next.bid == 0 {
            #[cfg(test)]
            assert_eq!(next.dev, devbox.dev, "{next:?} and {devbox:?} are not on the same device");
            // if next box is free, coleacse
            match devbox.dev {
                Dev::Cuda => &mut self.gpu_free[shard_id],
                Dev::Host => &mut self.cpu_free[shard_id],
            }.remove(&(next.len, next));
            devbox.len += next.len;
            devbox.nxt = next.nxt;
        }
        let last = unsafe{&mut *devbox.lst};
        if last.bid == 0 {
            #[cfg(test)]
            assert_eq!(last.dev, devbox.dev, "{last:?} and {devbox:?} are not on the same device");
            // if last box is free, coleacse
            match devbox.dev {
                Dev::Cuda => &mut self.gpu_free[shard_id],
                Dev::Host => &mut self.cpu_free[shard_id],
            }.remove(&(last.len, next));
            devbox.len += last.len;
            devbox.ptr = last.ptr;
            devbox.lst = last.lst;
        }
        Ok(())
    }
    /// split a box to given size
    fn split(&mut self, devbox: &mut DevBox, size: usize) -> Result<(), DevErr> {
        unimplemented!("split box");
    }
    /// delete devbox from box id
    pub(super)
    fn del_box(&mut self, bid: usize) -> Result<(), DevErr> {
        // shard id
        let shard_id = bid & 0xffusize;
        // the shard where box lives
        let shard = &mut self.mem_used[shard_id];
        // remove the box from shard
        let devbox = match shard.remove(&(bid >> 8usize)) {
            Some(pdevbox) =>unsafe{&mut *pdevbox} ,
            None => Err(DevErr::BoxNotFound)?
        };
        // coalesce with last box
        self.merge(devbox)?;
        // set devbox's bid to 0, which means it's free now
        devbox.bid = 0usize;
        // add box to free list
        match devbox.dev {
            Dev::Cuda => &mut self.gpu_free[shard_id],
            Dev::Host => &mut self.cpu_free[shard_id],
        }.insert((devbox.len, devbox));
        Ok(())
    }
    /// move box to target device
    pub(super)
    fn mov_box(&mut self, devbox: &mut DevBox, dev: Dev) {
        unimplemented!("move box");
    }
    /// new devbox from box size
    pub(super)
    fn new_box(size: usize) -> Result<(), DevErr> {
        unimplemented!("new box");
    }
    /// copy devbox from another box
    pub(super)
    fn cpy_box(size: usize) -> Result<(), DevErr> {
        unimplemented!("copy box");
    }
}