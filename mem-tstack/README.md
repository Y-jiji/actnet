# mem-tstack

## state of this crate

| ---         | state   | description                  |
| ----------- | ------- | ---------------------------- |
| code        | pending | waiting device-ext           |
| design idea | working | already have some rough idea |
|             |         |                              |

## rough design idea

In deep learning oriented system optimization, there are already many runtime-based solutions of memory optimization, as [Y-jiji](https://github.com/Y-jiji) mentioned in [mem-tgraph](../mem-tgraph/README.md) . However, memory fragmentation problem are seemly unavoidable in these cases. However, if we know all operators in the whole forward-backward process, things will be much easier: instead of releasing-on-need strategy, we can determine what tensor we will release, and allocate them in temporary locations. 

For example, suppose `B` is the cheapest tensor to release when `D` comes and `A` is the second cheapest (although this is almost unlikely). 

````rust
================
[ DTR APPROACH ]
================

T = S + 1 :: A -> B
-------------------------------------------------------------------------------------
... |       A       |       B       |                                               |
-------------------------------------------------------------------------------------
T = S + 2 :: (A, B) -> C
-------------------------------------------------------------------------------------
... |       A       |       B       |              C              |                 |
-------------------------------------------------------------------------------------
T = S + 3 :: load D, size(D) = size(C), OOM!
-------------------------------------------------------------------------------------
... |       A       |       B       |              C              |                 |
-------------------------------------------------------------------------------------
T = S + 4 :: release B, release C, place D
-------------------------------------------------------------------------------------
... |       A       |              D              |                                 |
-------------------------------------------------------------------------------------
T = S + 5 :: computation depend on C and D; rematerialize C; C depends on B, rematerialize B ...
````

However, if the memory manager is aware of future operations, there might be better approaches that minimize memory segement. The basic idea is: for forward computation, if we will release some tensor on some computation step, we should place it **after** where we want to store our computation result. 


````rust
===================
[ BETTER APPROACH ]
===================

T = S + 1 :: A -> B
-------------------------------------------------------------------------------------
... |       A       |                             |       B       |                 |
-------------------------------------------------------------------------------------
T = S + 2 :: (A, B) -> C
-------------------------------------------------------------------------------------
... |       A       |              C              |       B       |                 |
-------------------------------------------------------------------------------------
T = S + 3 :: load D, size(D) = size(C), OOM!
-------------------------------------------------------------------------------------
... |       A       |              C              |       B       |                 |
-------------------------------------------------------------------------------------
T = S + 4 :: release B, place D
-------------------------------------------------------------------------------------
... |       A       |              C              |              D              |
-------------------------------------------------------------------------------------
T = S + 5 :: compute in place of D, may be
-------------------------------------------------------------------------------------
... |       A       |              C              |              E              |
-------------------------------------------------------------------------------------
````

For backward propagation, it is much easier, generally most tensors are released and moved back to host memory, their will be no temporary offloadings. 

## ref this page

