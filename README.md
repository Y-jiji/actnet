# Actnet
An neural network toolkit in pure rust. 

# General Architecture

```
======================================================================================
       layer | functionality
--------------------------------------------------------------------------------------
      device | unify computation runtime API, implement operators
             | implement distributed computation  
--------------------------------------------------------------------------------------
     ndarray | utilize device layer, implement ndarray operations declaratively
             | this data structure is supposed to be immutable
--------------------------------------------------------------------------------------
      tensor | implement backward hooks with mutable data, utilizing ndarray
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      neuron | implement autograd functions without inner state
--------------------------------------------------------------------------------------
      neural | implement neuron with inner states
     network | implement common neural network building blocks, e.g. SelfAttention
 - - - - - - | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        data | implement data loading and preprocessing tools
     utility | 
======================================================================================
```

