# Stream Model for High Performance Computation

In this crate, the concept `device` refers to a stream computation model. 

A `stream computation model` (SCM) is a subclass of [actor-model](https://en.wikipedia.org/wiki/Actor_model). 

To be specific, it receives the following types of messages: 
```
NewBox(box, val) : register a new box with value
DeleteBox(box) : delete a box
Launch(operator, read_box_0, read_box_1, ... read_box_n, write_box) : launch an operator on stream, where an operator is a declarative operator. 
Terminate : terminate this stream, returns inner state
```
And maintains at least one inner state:
```
BoxValMap : a mapping from box to value
```

It also satisfies a special constraint, called `Weak FIFO` property, which we will define in the following subsection in terms of `standarded stream` . 

## Standard Stream

An `standard stream` is a stream that executes operators in FIFO order. 

```pseudocode
map = INIT_EMPTY_BOX_VAL_MAP()
LOOP {
    MATCH RECVIEVE_MESSAGE() {
        Terminate => BREAK,
        NewBox(box, val) => map[box] = val,
        DeleteBox(box) => map[box] = NULL,
        Launch(operator, read_box_0, read_box_1, ... read_box_n, write_box) => 
            map[write_box] = operator(read_box_0, read_box_1, ... read_box_n),
    }
}
RETURN map
```

## Weak FIFO Property

Weak FIFO property means for every message sequence that ends with `Terminate` and contains no `Terminate`, the actor returns the same `map` as the `standard stream`. 

The only difference operator reordering is permitted, as far as it doesn't change the termination state. 

# The Shadow Memory Technic

Given the computation model above, it is possible to use a special technic to optimize asynchronous launchs. 

## A CUDA Stream Example

TODO@Y-jiji(easy desc, also about it's goodness in modular design)

## Take Error State into Consideration

TODO@Y-jiji(slow shadow, checkpoints)

