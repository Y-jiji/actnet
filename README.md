# Actnet
An neural network toolkit in pure rust. 

# General Architecture

```
======================================================================================
       layer | functionality
--------------------------------------------------------------------------------------
      device | unify computation runtime API, implement operators
             | in future: implement distributed computation
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         mem | a memory management schedulor toolkit
             | some components will have unified api
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
  - [ ] device-cuda
  - [ ] device-toy
- [ ] mem
  - [x] mem-slotvec
  - [x] mem-allocator
  - [ ] mem-tgraph
- [ ] ndarray
- [ ] tensor
- [ ] neuron
- [ ] nn
- [ ] util

# State of this project

This project has only one developer [Y-jiji](https://github.com/Y-jiji). 

Currently Y-jiji is a senior undergraduate student at [DaSE -- ECNU](https://www.ecnu.edu.cn/wzcd/xxgk/yxsz.htm). 

There are many detailed ideas in his head, he just needs some time to type them out. 

However, curriculum projects are sometimes really heavy (20+ projects in a single semester! cheers!) and he cannot just f\*\*\* them off because a good GPA is very critical to his pursuit of further academic study. 
