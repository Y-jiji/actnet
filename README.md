# Project Proposal

Actnet aims to represent different memory accessing patterns in a unified way and provide high-level optimizations for all of them. 

Previously, convolution, self-attention, batch-norm and usual matrix multiplication are all handcraftedly optimized, which is not labor-efficient, and only focus on optimality of each computation steps. The main reason people doing this is the huge optimzation space to explore. 

However, I reckon that if computation were formalized properly, production-ready code could be discovered in a reasonable time, even in distributed computation cases. 

# Codebase Walkthrough

```
---------------+--------------------------------------------------------------------------------------
    ir-linajbr | an extended map-reduce-based ir
        (core) | try to unify all GeMM patterns! 🤯🤯🤯🤯🤯
               +----------+---------------------------------------------------------------------------
               | syntax   | interface syntax definition 
               |          | a easy-reading syntax for your sanity! 🥹🥹🥹🥹🥹
               |          | status: OK
               +----------+---------------------------------------------------------------------------
               | denaming | name resolution or fresh-naming ---- the same thing
               |          | mundane 🥱🥱🥱🥱🥱
               |          | status: TODO
               +----------+---------------------------------------------------------------------------
               | dataflow | dataflow analysis for auto parallelism and global optimization
               |          | and ... model device memory with ownership! 😍😍😍😍😍
               |          | status: TODO
               +----------+---------------------------------------------------------------------------
               | typeck   | type check and type inference with good defaults
               |          | you eliminate errors, or i eliminate your code! 💣💣💣💣💣
               |          | status: TODO
---------------+--------------------------------------------------------------------------------------
    ir-linajbr | the side-car crate for ir-linajbr (core) that embeds your code in rust
       (macro) | using macros is good but writing them is mind-blowing .....
               | status: OK
---------------+--------------------------------------------------------------------------------------
        rt-<x> | various run time backends
               | feed them data and ir-linajbr! 🤤🤤🤤🤤🤤
               +---------------+----------------------------------------------------------------------
               | <x>=cuda      | a verrryyy good runtime wrapper for cuda driver
               |               | no more pointers like CUmodule and CUfunction! 🥵🥵🥵🥵🥵
               |               | status: TODO
               +---------------+----------------------------------------------------------------------
               | <x>=interpret | a verrryyy slow runtime directly runs ir-linajbr
               |               | for san-check! 🐙🐙🐙🐙🐙
               |               | status: TODO
               +---------------+----------------------------------------------------------------------
               | <x>=????????  | coming soon
---------------+---------------+----------------------------------------------------------------------
```