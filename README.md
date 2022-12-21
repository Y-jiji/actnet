# Caution

Contemporarily, this repository is experimental and under solo-development. 

Even the main branch may contain code that don't compile. 

# Actnet: Scaling out painlessly

Goal of actnet is to provide a general-purpose scientific computation toolkit. 

Actnet's abstraction for computation resources is unified at any level. The idea comes from the following analogy: 

| Single Machine                                      | Actnet                                  |
| --------------------------------------------------- | --------------------------------------- |
| threads                                             | actors                                  |
| SIMD on a single CPU core                           | SIMT computation on a single actor node |
| launching thread level map reduce with thread pools | coordinating actors as a bundle actor   |
| pipelining classic five-stage dataflow              | pipelining multi-GPU dataflow           |
| shared RAM                                          | seperated RAM                           |

The only difference is where data lives, which is the problem actnet is trying to tackle. 

# Architecture

Gerernally speaking, this repository have an LLVM-like project structure. 

Developers can freely combine crates in this project to build softwares (crate: smallest unit from compiler's viewpoint). 

```
======================================================================================
       layer | functionality
--------------------------------------------------------------------------------------
          dv | [d]e[v]ice
             | unify computation runtime API, stream executors
             | in future: implement distributed computation
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          mm | [m]emory [m]anagement
             | memory allocation & memory hotspot accounting tools
             | memory mangement extension for specific workload
--------------------------------------------------------------------------------------
        sctk | [s]cientific [c]omputing [t]ool[k]it
             | utilize device layer, implement unified interface for arbitary device
             | implement classic machine learning algorithms
--------------------------------------------------------------------------------------
        util | [util]ity for data preprocessing
             | implement data loading and preprocessing tools
======================================================================================
```

# TODO list

- [ ] phase-1: start from trivial tasks

  - design dv-api, specify related concepts

  - implement dv-toy for testing, get familar with no-parallel numerical linear algebra algorithms

  - implement ndarray, modify dv-api for further applications
  - implement a single-stream synchronized version dv-cuda
  - implement a multi-stream version dv-cuda
  - extra work: classic machine learning algorithms, etc. 

- [ ] phase-2: optimize neural network on a single machine

  - implement a inter-thread bridge (dv-bridge-thrd)

  - implement a memory-seperated device for testing (dv-bundle-thrd)

  - implement autograd operators, sctk-tensor, sctk-neural-net
  - implement numerical gradient for sanity check
  - implement DTR for dynamic back-propagation workload
  - implement 'record and compile' for static back-propagation workload (mm-tstack)
  - extra work: common neural network blocks, data loading utilities, etc. 

- [ ] phase-3: distribute computation on multiple machines

  - implement a map-reduce-like system
  - design failure tolerance mechanisms with model checking (dv-bridge-proc)
  - try to reduce communication overhead in model parllelism
  - implement a data structure that captures data dependencies (mm-tgraph)

  - implement a dv-bundle that coordinates multiple devices (dv-bundle-proc)
  - extra work: direct data fetching from database, nvlink for direct memcpy, zero-copy data moving, etc. 

- [ ] phase-4: join various devices, various workloads and various data resources
  - extend to OpenCL-based, DPCPP-based, ... , and ACL-based platforms
  - memory management for simulation workload, maybe
  - blockchain cloud computing, or quantum computing, maybe
  - direct connection support for datalakes and databases, maybe

# State of this project

This project has only one developer [Y-jiji](https://github.com/Y-jiji). 

Currently Y-jiji is a senior undergraduate student at [DaSE -- ECNU](https://www.ecnu.edu.cn/wzcd/xxgk/yxsz.htm). 

There are many detailed ideas in his head, he just needs some time to type them out. 

However, curriculum projects are sometimes really heavy (20+ projects in a single semester! cheers!) and he cannot just f\*\*\* them off because a good GPA is very critical to his pursuit of further academic study. 
