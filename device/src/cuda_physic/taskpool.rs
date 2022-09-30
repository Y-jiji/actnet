use std::marker::PhantomPinned;
use std::thread;

#[derive(Clone, Debug, PartialEq)]
pub(super)
struct TaskPool<const SIZE: usize> {
    /// 1K Bytes for signal slots
    signal: [u8; SIZE],
    /// start offset
    offset: usize,
    /// used length,
    uselen: usize,
    /// signal should be pinned in memory
    pinpin: PhantomPinned,
    /// thread id for local thread
    thrdid: thread::ThreadId,
}

impl<const SIZE_BYTE: usize> TaskPool<SIZE_BYTE> {
    const SIZE_BIT: usize = SIZE_BYTE * 8;
    pub(super)
    fn new() -> Self {
        TaskPool {
            signal: [0u8; SIZE_BYTE],
            offset: 0, uselen: 0, 
            thrdid: thread::current().id(),
            pinpin: PhantomPinned, 
        }
    }
    /// get an empty task
    pub(super)
    fn get(&mut self) -> Result<usize, &'static str> {
        if thread::current().id() != self.thrdid
        { panic!("TaskPool: should not call method 'get' if taskpool isn't initialized in this thread"); }
        let task = (self.offset + self.uselen) & (Self::SIZE_BIT-1);
        if self.uselen == Self::SIZE_BIT
        { return Err("TaskPool: no signal slot"); }
        self.uselen += 1;
        return Ok(task);
    }
    /// set self.signal[task] to 1
    pub(super)
    fn put(&mut self, task: usize) {
        self.signal[task >> 3] |= 1 << (task & 0x7);
    }
    /// return the finished task segement, put these task bits back to pool
    pub(super)
    fn ack(&mut self) -> (usize, usize) {
        if thread::current().id() != self.thrdid
        { panic!("TaskPool: should not call method 'ack' if taskpool isn't initialized in this thread"); }
        let mut ret = (self.offset, self.offset);
        while self.signal[self.offset >> 3] == u8::MAX {
            self.signal[self.offset >> 3] = 0u8;
            ret.1 = self.offset + 8;
            self.offset += 8;
            self.offset &= Self::SIZE_BIT - 1;
            self.uselen -= 8;
        }
        while self.signal[self.offset >> 3] & (1 << (self.offset & 0x7)) != 0 {
            self.signal[self.offset >> 3] ^= 1 << (self.offset & 0x7);
            ret.1 = self.offset + 1;
            self.offset += 1;
            self.offset &= Self::SIZE_BIT - 1;
            self.uselen -= 1;
        }
        return ret;
    }
    /// print the task pool
    #[cfg(test)]
    pub(super)
    fn print(&self) {
        let each_row = 16;
        let sep = "=".repeat(each_row*8 + each_row-1);
        println!("{sep}");
        for i in 0..(1024/each_row) {
            for j in 0..each_row {
                for k in 0..8 {
                    print!("{}", (self.signal[i*each_row+j] >> k) & 1);
                }
                print!(" ");
            }
            print!("\n");
        }
        println!("{sep}");
    }
}

#[cfg(test)]
mod test {
    use super::TaskPool;
    use std::thread;
    use std::sync::Arc;
    use parking_lot::Mutex;
    #[test]
    fn test_task_pool() {
        /* use taskpool in parallelized program */
        let x_parallel = Arc::new(Mutex::new(TaskPool::<1024>::new()));
        let mut handle = Vec::new();
        for _ in 0..2019 {
            let x_clone = Arc::clone(&x_parallel);
            handle.push(thread::spawn(move || {
                for _ in 0..21 {
                    let mut x = x_clone.lock();
                    loop {if let Ok(tid) = x.get() {
                        x.put(tid);
                        break;
                    } thread::yield_now();}
                }
                let mut x = x_clone.lock();
                x.ack();
            }));
        }
        for h in handle.into_iter() {
            h.join().unwrap();
        }
        /* use taskpool in local thread */
        let mut x = TaskPool::<1024>::new();
        for _ in 0..2019 {
            for _ in 0..21 {
                let tid = x.get().unwrap();
                x.put(tid);
                x.ack();
            }
        }
        /* sanity check */
        assert_eq!(x, x_parallel.lock().to_owned());
    }
}