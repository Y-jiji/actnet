use super::*;
use std::alloc::*;

struct MemNode {
    // base address for memory segement
    addr: *mut Void,
    // size of the memory segement
    size: usize,
    // pointers for free list
    l_prev: *mut MemNode,
    l_succ: *mut MemNode,
    // pointers for physical memory list
    m_prev: *mut MemNode,
    m_succ: *mut MemNode,
}

impl MemNode {
    fn new(addr: *mut Void, size: usize) -> *mut MemNode {
        let node = unsafe{&mut*(alloc(Layout::new::<MemNode>()) as *mut _)};
        *node = MemNode {addr, size,
            l_prev: node, l_succ: node,
            m_prev: node, m_succ: node};
        node
    }
}

struct FreeList {
    head: Vec<*mut MemNode>,
    rate: f64,
    smallest: usize,
}

impl Drop for FreeList {
    fn drop(&mut self) {
        // dealloc heads, the ownership of physically meaningful memory nodes are managed by physical list
        for x in &self.head {
            unsafe{dealloc(*x as *mut u8, Layout::new::<FreeList>())};
        }
    }
}

impl FreeList {
    const MASK: usize = usize::MAX << 1;
    fn new(size: usize, rate: f64, smallest: usize) -> Self {
        assert!(size % 2 == 0);
        assert!(size >= smallest);
        let log = f64::log((size / smallest) as f64, rate) as usize;
        let mut head = Vec::new();
        for _ in 0..log {
            head.push(MemNode::new(null_mut(), 0));
        }
        return Self { head, rate, smallest }
    }
    fn index(&self, size: usize) -> usize {
        let log = f64::log(size as f64 / self.smallest as f64, self.rate);
        let index = usize::max(0, usize::min(log as usize, self.head.len() - 1));
        return index;
    }
    /// insert node into free list
    fn insert(&mut self, node: *mut MemNode) {
        // set the mask as free
        // println!("free list insert {:?}", unsafe{(*node).addr});
        unsafe{(*node).size &= Self::MASK};
        let i = self.index(unsafe{(*node).size});
        let head = self.head[i];
        // TODO@Y-jiji(sort pointers into ascending address order)
        unsafe {
            // head <- node -> succ
            (*node).l_prev = head;
            (*node).l_succ = (*head).l_succ;
            // head -> node <- succ
            (*(*head).l_succ).l_prev = node;
            (*head).l_succ = node;
        }
    }
    /// remove node from free list
    fn remove(&mut self, node: *mut MemNode) {unsafe{
        #[cfg(test)] {
            assert!(!node.is_null());
            for head in &self.head 
            { assert!(*head != node); }
        }
        (*node).size |= 1;
        let l_prev = (*node).l_prev;
        let l_succ = (*node).l_succ;
        // l_prev <-> l_succ
        (*l_prev).l_succ = l_succ;
        (*l_succ).l_prev = l_prev;
        // node <-> node
        (*node).l_prev = node;
        (*node).l_succ = node;
    }}
    /// find a node greater or equal to size
    fn find_ge(&mut self, size: usize) -> *mut MemNode {
        let size = usize::max(size, self.smallest);
        for i in self.index(size)..self.head.len() {unsafe{
            let head = self.head[i];
            let mut node = (*head).l_prev;
            while node != head {
                if (*node).size >= size
                { return node; }
                else
                { node = (*node).l_prev; }
            }
        }}
        return null_mut();
    }
    /// print debug info
    #[cfg(test)]
    fn debug_print(&self) -> String {
        let mut return_string = String::new();
        for head in &self.head {
            let head = *head;
            let mut ptr = unsafe{(*head).l_prev};
            while ptr != head {
                unsafe{assert!((*ptr).size & 1 == 0);}
                return_string += &format!("-> {:?} ", unsafe{(*ptr).addr});
                ptr = unsafe{(*ptr).l_prev};
                unsafe{assert!((*(*ptr).l_prev).l_succ == ptr);}
            }
            return_string += "\n"
        }
        return_string
    }
    /// a sanity check when memory segment are ought to be a whole
    #[cfg(test)]
    fn san_check(&self, lbound: *mut Void, rbound: *mut Void) {
        let x = (*self.head).last().unwrap();
        let x = unsafe{(**x).l_succ};
        unsafe{
            assert!((*x).addr == lbound);
            assert!((*x).addr.add((*x).size) == rbound);
        }
    }
}

struct PhysList {
    addr_info: HashMap<*mut Void, *mut MemNode>,
    smallest: usize,
    lbound: *mut Void,
    rbound: *mut Void,
}

impl Drop for PhysList {
    /// drop each every node recorded in address info
    fn drop(&mut self) {
        for x in self.addr_info.values() {
            unsafe{dealloc(*x as *mut u8, Layout::new::<MemNode>())};
        }
    }
}

impl PhysList {
    const MASK: usize = usize::MAX << 1;
    fn new(lbound: *mut Void, rbound: *mut Void, smallest: usize) -> Self {
        let mut addr_info = HashMap::new();
        addr_info.insert(lbound, MemNode::new(lbound, unsafe{rbound.offset_from(lbound)} as usize));
        return Self { lbound, rbound, smallest, addr_info }
    }
    /// split a node
    fn split(&mut self, node: *mut MemNode, size: usize) -> (*mut MemNode, *mut MemNode) {unsafe{
        // satisfy align requirement(keep size greater than size)
        assert!(size & Self::MASK == size);
        // if splitting is bad
        if size < self.smallest || ((*node).size & Self::MASK) - size < self.smallest { 
            return (node, null_mut());
        }
        // split node into left half and right half
        let (lhalf, rhalf) = 
            (node, MemNode::new((*node).addr.add(size), ((*node).size & Self::MASK) - size));
        // adapt node's size
        (*node).size = size | ((*node).size & 1);
        // lhalf <- rhalf -> m_succ
        (*rhalf).m_prev = lhalf;
        (*rhalf).m_succ = (*lhalf).m_succ;
        // lhalf -> rhalf <- m_succ
        (*(*rhalf).m_prev).m_succ = rhalf;
        (*(*rhalf).m_succ).m_prev = rhalf;
        // register rhalf to address info
        self.addr_info.insert((*rhalf).addr, rhalf);
        return (lhalf, rhalf);
    }}
    /// whether the segement is the last one
    fn is_rbound(&self, node: *mut MemNode) -> bool {unsafe{
        (*node).addr.add((*node).size & Self::MASK) == self.rbound
    }}
    /// whether the segement is the first one
    fn is_lbound(&self, node: *mut MemNode) -> bool {unsafe{
        (*node).addr == self.lbound
    }}
    /// merge this node and its adjacently next node
    fn merge(&mut self, node: *mut MemNode) {unsafe{
        // println!("merge {:?}", (*node).addr);
        // remove address from address infomation
        self.addr_info.remove(&(*(*node).m_succ).addr);
        // a m_succ var for easy access
        let m_succ = (*node).m_succ;
        // extend size to cover the m_succ
        (*node).size += (*m_succ).size & Self::MASK;
        // assert m_succ is right to node
        assert!(0 < (*m_succ).addr.offset_from((*node).addr));
        // delete m_succ from physical list
        (*(*m_succ).m_succ).m_prev = node;
        (*node).m_succ = (*m_succ).m_succ;
        // dealloc m_succ
        dealloc(m_succ as *mut _, Layout::new::<MemNode>());
    }}
    /// print debug info about list
    #[cfg(test)]
    fn debug_print_list(&self) -> String {unsafe{
        let mut return_string = String::new();
        let head = *self.addr_info.get(&self.lbound).unwrap();
        return_string += &format!("-> {:?}|{:?} ", (*head).addr, (*head).size & 1);
        let mut ptr = (*head).m_succ;
        while ptr != head {
            return_string += &format!("-> {:?}|{:?} ", (*ptr).addr, (*ptr).size & 1);
            ptr = (*ptr).m_succ;
        }
        return_string
    }}
    /// print debug info about memory usage
    #[cfg(test)]
    fn debug_print_free(&self) -> String {
        // create a masking array, where '#' represents free space, ' ' represents used space
        let mut return_array = ['#'; 32*128];
        // project a pointer to array index
        let project = |ptr: *mut Void|-> usize {
            let total = unsafe{self.rbound.offset_from(self.lbound)} as usize;
            let offset = unsafe{ptr.offset_from(self.lbound)} as usize;
            32*128*offset / total
        };
        // modify the masking array
        for (ptr, node) in self.addr_info.iter() {
            if unsafe{&(**node).size} & 1 == 0 { continue; }
            let (lower, upper) = (*ptr, unsafe{(*ptr).add((**node).size & Self::MASK)});
            for i in project(lower)..project(upper) {
                return_array[i] = ' ';
            }
        }
        // convert the masking array into string
        let mut return_string = String::new();
        for chunk in return_array.chunks(128) {
            return_string += &String::from_iter(chunk);
            return_string += "\n";
        }
        return return_string;
    }
    #[cfg(test)]
    fn san_check(&self) {
        for (ptr, node) in self.addr_info.iter() {unsafe{
            let node = *node;
            let ptr = *ptr;
            assert!((*node).addr == ptr);
        }}
    }
}

struct SimpleState {
    phys_list: PhysList,
    free_list: FreeList,
}

impl MemState for SimpleState {
    fn new(lbound: *mut Void, rbound: *mut Void) -> Self {
        // initialize physical list
        let phys_list = PhysList::new(lbound, rbound, 64);
        let size = unsafe{rbound.offset_from(lbound) as usize};
        let first_node = phys_list.addr_info.values().next().unwrap();
        // initialize free list
        let mut free_list = FreeList::new(size, 1.2, 64);
        free_list.insert(*first_node);
        Self { phys_list, free_list }
    }
    fn alloc(&mut self, size: usize) -> Result<*mut Void, MemErr> {
        // align size
        let size = (usize::max(size, self.phys_list.smallest) + 1) & PhysList::MASK;
        // try to find a large enough node
        let node = self.free_list.find_ge(size);
        // if there are is such node, raise error
        if node.is_null() { return Err(MemErr::OutOfMemory); }
        // remove node from free list
        self.free_list.remove(node);
        // split node into two halves
        let (lhalf, rhalf) = self.phys_list.split(node, size);
        // insert the right half into free list
        if !rhalf.is_null()
        { self.free_list.insert(rhalf); }
        return Ok(unsafe {(*lhalf).addr});
    }
    fn free(&mut self, ptr: *mut Void) -> Result<(), MemErr> {
        if ptr.is_null() { Err(MemErr::InvalidPtr)?; }
        // get the node from address pointer
        let mut node = match self.phys_list.addr_info.get(&ptr) {
            Some(x) => *x,
            None => Err(MemErr::InvalidPtr)?,
        };
        // println!("free {ptr:?}");
        unsafe{assert!(ptr == (*node).addr);}
        if !self.phys_list.is_lbound(node)
        && unsafe {(*(*node).m_prev).size & 1 == 0} {
            // is left bound
            let is_rbound = self.phys_list.is_rbound(node);
            // merge previous node with this node
            let prev = unsafe{(*node).m_prev};
            // remove node from free list
            self.free_list.remove(prev);
            // merge current node with next
            self.phys_list.merge(prev);
            assert!(self.phys_list.is_rbound(prev) == is_rbound);
            node = prev;
        }
        if !self.phys_list.is_rbound(node)
        && unsafe {(*(*node).m_succ).size & 1 == 0} {
            // remove node from free list
            self.free_list.remove(unsafe{(*node).m_succ});
            // merge current node with next
            self.phys_list.merge(node);
        }
        // insert node back into free list
        self.free_list.insert(node);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::random;

    #[test]
    fn test_free_list() {
        let mut free_list = FreeList::new(1 << 20, 1.5, 64);
        let mut vec_allocated = Vec::new();
        println!("-----------test free list-------------");
        for _ in 0..100 {
            println!("======================================");
            let size = random::<usize>() % 100000;
            let ptr = unsafe{null_mut::<Void>().add(random::<usize>() % 100)};
            println!("insert ({ptr:?}, {size})");
            let node = MemNode::new(ptr, size);
            vec_allocated.push(node);
            free_list.insert(node);
            println!("--------------------------------------");
            print!("{}", free_list.debug_print());
        }
        for _ in 0..100 {
            println!("======================================");
            let node = vec_allocated.pop().unwrap();
            println!("remove ({:?}, {})", unsafe{(*node).addr}, unsafe{(*node).size});
            free_list.remove(node);
            println!("--------------------------------------");
            print!("{}", free_list.debug_print());
            println!("--------------------------------------");
        }
    }

    #[test]
    fn test_phys_list() {unsafe{
        const MEM_SIZE: usize = 10000;
        let lbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000);
        let rbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000 + MEM_SIZE);
        let mut phys_list = PhysList::new(lbound, rbound, 64);
        for _ in 0..20 {
            let node = phys_list.addr_info.get(&lbound).unwrap();
            let (lhalf, rhalf) = phys_list.split(*node, ((**node).size >> 2) << 1);
            if !rhalf.is_null() {
                assert!((*lhalf).m_succ == rhalf);
                assert!((*rhalf).m_prev == lhalf);
            }
            println!("--------------------------------------");
            println!("{}", phys_list.debug_print_list());
        }
        for _ in 0..20 {
            let node = *phys_list.addr_info.get(&lbound).unwrap();
            if (*node).m_prev != node 
            { phys_list.merge(node); }
            println!("--------------------------------------");
            println!("{}", phys_list.debug_print_list());
        }
        phys_list.san_check();
        let node = *phys_list.addr_info.get(&lbound).unwrap();
        assert!((*node).addr == lbound);
        assert!((*node).addr.add((*node).size) == rbound);
    }}

    #[test]
    fn test_init_sanity() {unsafe{
        const MEM_SIZE: usize = 10000;
        let lbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000);
        let rbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000 + MEM_SIZE);
        println!("{:?}", lbound);
        let simple_state = SimpleState::new(lbound, rbound);
        println!("--------------------------------------");
        print!("{}", simple_state.free_list.debug_print());
        println!("--------------------------------------");
        simple_state.free_list.san_check(lbound, rbound);
    }}

    #[test]
    fn test_rand_alloc() {unsafe{
        const TEST_CNT: usize = 100000;
        const RAND_BOND: usize = 1562;
        const MEM_SIZE: usize = 1000000;
        let lbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000);
        let rbound = null_mut::<Void>().add((random::<usize>() << 1) % 1000 + MEM_SIZE);
        let mut simple_state = SimpleState::new(lbound, rbound);
        let mut err_cnt = 0;
        let mut not_free = HashMap::new();
        let mut max_occ = 0;
        let mut occ = 0;
        for i in 0..TEST_CNT {
            if i % (TEST_CNT / 10) == 0 { println!("{i}/{TEST_CNT}"); }
            let size = random::<usize>() % 1000 + 64;
            let ptr = match simple_state.alloc(size) {
                Ok(p) => p,
                Err(_) => {err_cnt += 1; continue;},
            };
            occ += size;
            // simple_state.phys_list.san_check();
            // println!("======================================");
            // println!("alloc {:?}", ptr);
            // println!("--------------------------------------");
            // println!("free list");
            // print!("{}", simple_state.free_list.debug_print());
            // println!("--------------------------------------");
            // println!("phys list");
            // println!("{}", simple_state.phys_list.debug_print_list());
            // println!("======================================");
            not_free.insert(ptr, size);
            let free_ptr = not_free.keys().nth(random::<usize>() % RAND_BOND);
            if let Some(free_ptr) = free_ptr {
                let free_ptr = *free_ptr;
                let size = not_free.get(&free_ptr).unwrap();
                // println!("======================================");
                // println!("free {:?}", free_ptr);
                simple_state.free(free_ptr).unwrap();
                occ -= size;
                // simple_state.phys_list.san_check();
                not_free.remove(&free_ptr);
                // println!("--------------------------------------");
                // println!("free list");
                // print!("{}", simple_state.free_list.debug_print());
                // println!("--------------------------------------");
                // println!("phys list");
                // println!("{}", simple_state.phys_list.debug_print_list());
                // println!("======================================");
            }
            max_occ = usize::max(occ, max_occ);
        }
        for free_ptr in not_free.keys() {
            // println!("======================================");
            // println!("free {:?}", free_ptr);
            simple_state.free(*free_ptr).unwrap();
            // println!("--------------------------------------");
            // print!("{}", simple_state.free_list.debug_print());
            // println!("--------------------------------------");
            // println!("{}", simple_state.phys_list.debug_print_list());
            // println!("======================================");
        }
        not_free.clear();
        // println!("======================================");
        // print!("{}", simple_state.free_list.debug_print());
        // println!("--------------------------------------");
        // println!("{}", simple_state.phys_list.debug_print_list());
        // println!("======================================");
        // println!("({lbound:?}, {rbound:?})");
        simple_state.free_list.san_check(lbound, rbound);
        println!("error count: {err_cnt}/{TEST_CNT}");
        println!("max utilized space: {max_occ}/{MEM_SIZE}");
        simple_state.phys_list.debug_print_free();
    }} 
}