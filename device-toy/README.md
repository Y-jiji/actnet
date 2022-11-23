# device-toy

## Goal

Goal of this crate is to provide a standard stream computation model for testing. 

Whatever complex hardware they utilize or scheduling algorithms they use, stream computation models (trait Device) should be equivalent to device-toy in the sense of weak stream equivalence. 

## What is "weak stream equivalence"?

First I would like to introduce what is a stream computation model, or stream in short. (This is a term coined by Y-jiji)

For a stream computation model, there are 3 possible operations. 

```
======================================================================================
        operation | description
--------------------------------------------------------------------------------------
    create symbol | create a symbol with given content
--------------------------------------------------------------------------------------
  retrieve symbol | delete a symbol, return its content
--------------------------------------------------------------------------------------
  launch function | eat some symbols, create a tuple of symbols with a given function
======================================================================================
```

For the sake of memory, we allow delete symbol without returning content. 

Symbols are assigned with an incremental number indicating the order of their creation. 

"Weak stream equivalence" is also a term coined by Y-jiji. 

It means for an arbitary state-determined operation sequence (i.e. sequence without randomization), when retrieving from the same variable, we should get exactly the same content. 

## Merging multiple streams

TODO@Y-jiji

## Why their is a 'kick' function in [device-api](../device-api/src/lib.rs)?

To achieve a better performance, a 'smart' stream may reorder its execution sequence. 

The most extreme way is being lazy, i.e. only starts computation when a 'retrieve symbol' operation arrives. 

For these streams, they may behave well working alone, but produce errors when merging multiple streams. 