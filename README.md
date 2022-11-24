# Actnet
An neural network toolkit in pure rust. 

# General Architecture

Gerernally speaking, this repository have an LLVM like project structure. 

Developers can freely combine sub-crates in this project to build their own projects (crate: syn. of library or binary in Rust). 

```
======================================================================================
       layer | functionality
--------------------------------------------------------------------------------------
      device | unify computation runtime API, implement operator executors
             | in future: implement distributed computation
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      bridge | make a device available to other threads / processes / machines
             | with a correspondent device client
             | in future: a master scheduler may collect a bundle of bridged devices 
             | and behaves like a single device
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         mem | memory management & accounting tools
             | in future: implement dynamic tensor rematerialization
--------------------------------------------------------------------------------------
     ndarray | utilize device layer, implement ndarray operations declaratively
             | this data structure is supposed to be immutable
--------------------------------------------------------------------------------------
      tensor | implement backward hooks with mutable data, utilizing ndarray
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      neuron | implement autograd functions without inner state
--------------------------------------------------------------------------------------
          nn | implement neuron with inner states
             | implement common neural network building blocks, e.g. Self-Attention
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        util | implement data loading and preprocessing tools
======================================================================================
```

# TODO list

- [ ] device
  - [ ] device-api
    - [ ] basic stream models
  
  - [ ] device-cuda
  - [x] device-toy
    - [ ] some operators are left unimplemented
  - [ ] device-ext
    - [ ] implement memory-management extensions
  
- [ ] mem
  - [x] mem-slotvec
  - [x] mem-allocator
  - [ ] mem-tgraph
  - [ ] mem-tstack
- [ ] ndarray
  - [x] display
  - [x] send operators to device
    - [ ] implement all operators
- [ ] tensor
  - [ ] display
  - [ ] implement using ndarray

- [ ] neuron
- [ ] nn
- [ ] util

# State of this project

This project has only one developer [Y-jiji](https://github.com/Y-jiji). 

Currently Y-jiji is a senior undergraduate student at [DaSE -- ECNU](https://www.ecnu.edu.cn/wzcd/xxgk/yxsz.htm). 

There are many detailed ideas in his head, he just needs some time to type them out. 

However, curriculum projects are sometimes really heavy (20+ projects in a single semester! cheers!) and he cannot just f\*\*\* them off because a good GPA is very critical to his pursuit of further academic study. 
