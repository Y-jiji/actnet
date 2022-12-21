pub trait MemManager {
    type Err;
    
    /// ### *description*
    /// pull a memory segement with size sz from managed memory pool
    /// ### *output*
    /// return usize as memory id, release implementation specific error on failure
    fn pull(&mut self, sz: usize) -> Result<usize, Self::Err> { todo!("pull({sz:?})") }

    /// ### *description*
    /// return a memory segement with memory id
    /// ### *output*
    /// return nothing, release implementation specific error on failure
    fn free(&mut self, n: usize) -> Result<(), Self::Err> { todo!("free({n:?})") }

    // 每收到一条消息, 都需要给出action(不能等待两条消息), 否则可能死锁. 
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}