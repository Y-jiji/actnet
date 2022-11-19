# TGraph

## Introduction

DTR (Dynamic Tensor Rematerialization) is a very useful memory management strategy in deep learning models. 

It allows training and using big models with limited GPU memory and reasonable throughput. 

The underlying mechanism concerns recomputing and reloading tensors. By recomputing and reloading tensors, some GPU memory can be released when memory is insufficient so that new tensors can readily take its space. 

The designing question here is which tensor (or tensors) should we release, so that it (or they) can be recomputed or reloaded at relatively low expense. 

## Goal of this crate

This crate aims to implement an accounting data structure (i.e. TGraph) for a DTR-like schedulor. 

## API

TODO(@Y-jiji)

## Refernce

[Dynamic Tensor Rematerialization](https://arxiv.org/abs/2006.09616)

[Dynamically Optimizing GPU Memory beyond Tensor Recomputation](https://arxiv.org/abs/2203.15980)
