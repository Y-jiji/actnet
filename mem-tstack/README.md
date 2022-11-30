# ext-tstack

## state of this crate

| ---         | state   | description                  |
| ----------- | ------- | ---------------------------- |
| code        | pending | waiting device-ext           |
| design idea | working | already have some rough idea |
|             |         |                              |

## a brief introduction to mm optimization in deep learning

DTR (Dynamic Tensor Rematerialization) is a very useful memory management strategy in deep learning models. 

It allows training and using big models with limited GPU memory and reasonable throughput. 

The underlying mechanism concerns recomputing and reloading tensors. By recomputing and reloading tensors, some GPU memory can be released when memory is insufficient so that new tensors can readily take its space. 

The designing question here is which tensor (or tensors) we should release, so that it (or they) can be recomputed or reloaded at relatively low expense. 

## DTR and DELTA

[Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)

[Dynamically Optimizing GPU Memory beyond Tensor Recomputation](https://arxiv.org/abs/2203.15980)


## rough design idea

In deep learning oriented system optimization, there are already many runtime-based solutions of memory optimization, as [Y-jiji](https://github.com/Y-jiji) mentioned in the previous section . However, memory fragmentation problem are seemly unavoidable in these cases. However, if we know all operators in the whole forward-backward process, things will be much easier: instead of releasing-on-need strategy, we can determine what tensor we will release, and allocate them in temporary locations. 

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

## ref this crate

@misc{actnet/ext-tstack,
  author = {Y-jiji},
  title = {GPU memory management policy for deep learning system load},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Y-jiji/actnet}}
}