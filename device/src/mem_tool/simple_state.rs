use super::*;

/// this is unsafely implemented, because there are no cross linked lists in rust
struct UnsafeListNode {
    list_last: *mut UnsafeListNode,
    list_next: *mut UnsafeListNode,
    phys_last: *mut UnsafeListNode,
    phys_next: *mut UnsafeListNode,
    phys_addr: *mut Void,
    phys_size: usize,
}

impl UnsafeListNode {
    fn new(phys_addr: *mut Void, phys_size: usize) -> *mut UnsafeListNode {
        let nptr = null_mut();
        let node = Box::leak(Box::new(
            UnsafeListNode{list_next: nptr, list_last: nptr, phys_last: nptr, phys_next: nptr, phys_addr, phys_size}));
        node.phys_last = node;
        node.phys_next = node;
        node.list_last = node;
        node.list_next = node;
        return node as *mut UnsafeListNode;
    }
    unsafe fn insert_as_list_next(this: *mut UnsafeListNode, next: *mut UnsafeListNode) -> *mut UnsafeListNode {
        (*next).list_next = (*this).list_next;
        (*next).list_last = this;
        (*(*this).list_next).list_last = next;
        (*this).list_next = next;
        return this;
    }
    unsafe fn insert_as_list_last(this: *mut UnsafeListNode, last: *mut UnsafeListNode) -> *mut UnsafeListNode {
        (*last).list_last = (*this).list_last;
        (*last).list_next = this;
        (*(*this).list_last).list_next = last;
        (*this).list_last = last;
        return this;
    }
    unsafe fn delete_as_list_node(this: *mut UnsafeListNode) -> *mut UnsafeListNode {
        (*(*this).list_next).list_last = (*this).list_last;
        (*(*this).list_last).list_next = (*this).list_next;
        (*this).list_last = this;
        (*this).list_next = this;
        return this;
    }
    unsafe fn insert_as_phys_next(this: *mut UnsafeListNode, next: *mut UnsafeListNode) -> *mut UnsafeListNode {
        (*next).phys_next = (*this).phys_next;
        (*next).phys_last = this;
        (*(*this).phys_next).phys_last = next;
        (*this).phys_next = next;
        return this;
    }
    unsafe fn insert_as_phys_last(this: *mut UnsafeListNode, last: *mut UnsafeListNode) -> *mut UnsafeListNode {
        (*last).phys_last = (*this).phys_last;
        (*last).phys_next = this;
        (*(*this).phys_last).phys_next = last;
        (*this).phys_last = last;
        return this;
    }
    unsafe fn delete_as_phys_node(this: *mut UnsafeListNode) {
        (*(*this).phys_next).phys_last = (*this).phys_last;
        (*(*this).phys_last).phys_next = (*this).phys_next;
        drop(Box::from_raw(this));
    }
    unsafe fn mark_as_used(this: *mut UnsafeListNode) {
        (*this).phys_size |= 1;
    }
    unsafe fn mark_as_free(this: *mut UnsafeListNode) {
        (*this).phys_size &= usize::MAX ^ 1;
    }
}

/// ALIGN mean align bits
pub(crate)
struct SimpleState<const ALIGN: usize> {
    /// memory segment it monitors
    lbound: *mut Void, rbound: *mut Void,
    /// the state of free memory
    free_list: Vec<*mut UnsafeListNode>,
    /// info
    info_map: HashMap<*mut Void, *mut UnsafeListNode>,
}

impl<const ALIGN: usize> SimpleState<ALIGN> {
    const ALIGN_MASK: usize = usize::MAX << ALIGN;
    const ALIGN_SIZE: usize = 1 << ALIGN;
    unsafe fn split_as_phys_node(&mut self, this: *mut UnsafeListNode, size: usize) -> (*mut UnsafeListNode, *mut UnsafeListNode) {
        let this_left = this;
        let this_right = UnsafeListNode::new((*this).phys_addr.add(size), (*this).phys_size - size);
        (*this).phys_size = (*this).phys_size - size;
        UnsafeListNode::insert_as_phys_next(this_left, this_right);
        self.info_map.insert((*this_right).phys_addr, this_right);
        return (this_left, this_right);
    }
    unsafe fn merge_as_phys_node(&mut self, this: *mut UnsafeListNode) -> *mut UnsafeListNode {
        let last = (*this).phys_last;
        let next = (*this).phys_next;
        let last_is_free = (*last).phys_size & 1 == 0;
        let next_is_free = (*next).phys_size & 1 == 0;
        if last_is_free {
            UnsafeListNode::delete_as_list_node(last);
            self.info_map.remove(&(*this).phys_addr);
            *self.info_map.get_mut(&(*last).phys_addr).unwrap() = this;
            (*this).phys_size += (*last).phys_size & Self::ALIGN_MASK;
            (*this).phys_addr = (*this).phys_addr.offset(-((*last).phys_size as isize));
            UnsafeListNode::delete_as_phys_node(last);
        }
        if next_is_free {
            UnsafeListNode::delete_as_list_node(next);
            self.info_map.remove(&(*next).phys_addr);
            (*this).phys_size += (*next).phys_size & Self::ALIGN_MASK;
            UnsafeListNode::delete_as_phys_node(next);
        }
        return this;
    }
    fn calc_which_free_list(&mut self, size: &usize) -> usize {
        let size_f = f64::from((size/Self::ALIGN_SIZE) as u32);
        let index = usize::min(size_f.log(1.1) as usize, self.free_list.len() - 1);
        return index;
    }
    unsafe fn get_from_free_list(&mut self, size: usize) -> *mut UnsafeListNode {
        let index = self.calc_which_free_list(&size);
        let mut list_ptr = null_mut();
        for i in index..self.free_list.len() {
            let list_head = self.free_list[i];
            list_ptr = (*list_head).list_next;
            while (*list_ptr).phys_size < size {
                list_ptr = (*list_ptr).list_next;
                if list_ptr == list_head {break;}
            }
            if list_ptr != list_head { return UnsafeListNode::delete_as_list_node(list_ptr); }
        }
        return null_mut();
    }
    unsafe fn put_into_free_list(&mut self, node: *mut UnsafeListNode) {
        let index = self.calc_which_free_list(&(*node).phys_size);
        UnsafeListNode::insert_as_list_next(self.free_list[index], node);
    }
    #[cfg(test)]
    // a good plotter for debugging
    fn print(&self) -> String {
        let mut population_vec = ['#'; 128*128];
        for x in self.info_map.values().into_iter() {
            if unsafe{(**x).phys_size} & 1 == 0 {continue;}
            let phys_size = unsafe{(**x).phys_size} & Self::ALIGN_MASK;
            let ptr = unsafe{(**x).phys_addr};
            assert!(unsafe{ptr.offset_from(self.rbound) as isize} < 0);
            assert!(unsafe{ptr.offset_from(self.rbound) + phys_size as isize} < 0);
            let istart = (128*128*unsafe{ptr.offset_from(self.lbound)}) / unsafe{self.rbound.offset_from(self.lbound)};
            let iend = (128*128*unsafe{ptr.offset_from(self.lbound) + phys_size as isize}) / unsafe{self.rbound.offset_from(self.lbound)};
            for i in istart..iend {
                population_vec[i as usize] = ' ';
            }
        }
        let mut population_str = String::new();
        for x in population_vec.chunks(128).into_iter() {
            population_str += &(*x).into_iter().collect::<String>();
            population_str += "\n";
        }
        population_str.pop();
        return population_str
    }
}

impl<const ALIGN: usize> MemState for SimpleState<ALIGN> {
    fn alloc(&mut self, size: usize) -> Result<*mut Void, MemErr> {
        let size = (size + Self::ALIGN_SIZE - 1) & Self::ALIGN_MASK;
        let node = unsafe{self.get_from_free_list(size)};
        if node == null_mut() { return Err(MemErr::OutOfMemory); }
        let (lhalf, rhalf) = unsafe{self.split_as_phys_node(node, size)};
        unsafe{UnsafeListNode::mark_as_used(lhalf)};
        unsafe{self.put_into_free_list(rhalf)};
        return Ok(unsafe{(*lhalf).phys_addr});
    }
    fn free(&mut self, ptr: *mut Void) -> Result<(), MemErr> {
        let node = match self.info_map.get(&ptr) {
            Some(x) => *x,
            None => return Err(MemErr::InvalidPtr),
        };
        let node = unsafe{self.merge_as_phys_node(node)};
        unsafe{self.put_into_free_list(node)};
        Ok(())
    }
    fn new(lbound: *mut Void, rbound: *mut Void) -> Self {
        let mut new_self = Self {
            lbound, rbound, free_list: Vec::new(),
            info_map: HashMap::new(),
        };
        let list_num = usize::max(
            1+f64::from((unsafe{rbound.offset_from(lbound)} as usize/Self::ALIGN_SIZE) as u32).log(1.1) as usize, 1);
        new_self.free_list.resize(
            list_num, UnsafeListNode::new(null_mut(), 0));
        let new_node = UnsafeListNode::new(lbound, unsafe{rbound.offset_from(lbound) as usize});
        new_self.info_map.insert(lbound, new_node);
        unsafe{new_self.put_into_free_list(new_node)};
        return new_self;
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use rand::random;
    use std::collections::HashSet;

    #[test]
    fn random_alloc() {
        let mut p_alloc = HashSet::new();
        let mut test_state = SimpleState::<5>::new(unsafe{null_mut::<Void>().add(10000)}, unsafe{null_mut::<Void>().offset(1000000)});
        let mut err_cnt = 0;
        // random alloc with random free
        for i in 0..10 {
            if i % 100 == 0 { println!("{i}/100000") }
            let x = random::<usize>() % 10000usize;
            let ptr = match test_state.alloc(x) {
                Ok(p) => p,
                Err(e) => {err_cnt += 1; continue;},
            };
            // println!("{:?}", ptr);
            p_alloc.insert(ptr);
            let ptr = match p_alloc.iter().nth(random::<usize>() & 0xff) {
                Some(x) => *x,
                None => {continue;}
            };
            if random::<usize>() & 0xf == 0 {
                match test_state.free(ptr) {
                    Ok(()) => {},
                    Err(e) => {err_cnt += 1}
                }
            }
            // println!("{}", ['-'; 128].into_iter().collect::<String>());
            // println!("{}", test_state.print());
            // println!("{}", ['-'; 128].into_iter().collect::<String>());
        }
        // print the state before free all
        println!("{}", ['-'; 128].into_iter().collect::<String>());
        println!("{}", test_state.print());
        println!("{}", ['-'; 128].into_iter().collect::<String>());
        // free the rest pointers
        for ptr in p_alloc.into_iter() {
            match test_state.free(ptr) {
                Ok(()) => {},
                Err(e) => {err_cnt += 1}
            };
        }
        // print the state after free all
        println!("{}", ['-'; 128].into_iter().collect::<String>());
        println!("{}", test_state.print());
        println!("{}", ['-'; 128].into_iter().collect::<String>());
        // print the final state
        println!("{err_cnt}/10000");
    }
}