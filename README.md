# actnet
An neural network toolkit in rust, implementation guided by actor-model. 

# to do list

- [ ] miscellaneous utilities
  - [ ] timer actor
  - [ ] heart-beat actor (for distributed applications)
  - [ ] logging actor

- [ ] device API layer
    - [ ] CUDA kernel functions
        - [ ] implement tensor dot
        - [ ] implement transpose
    - [ ] an actor API for a single CUDA stream locally
        - [ ] implement internal state
            - [ ] an ahead-of-time GPU memory layout state (the fast ShadowMem)
            - [ ] a post-time GPU memory layout state (the slow ShadowMem)
            - [ ] a sender_id pool for each job
        - [ ] implement message-receive
            - [ ] CUDA stream task finish message -> pop sender_id from pool -> reply
            - [ ] memory operation job -> change shadow layout -> send memory operation to device
            - [ ] kernel execution job -> add sender_id to pool ->  send kernel launch to device
            - [ ] timely check succeed -> update the slow ShadowMem
        - [ ] implement message-send
            - [ ] send kernel execution to CUDA stream
            - [ ] send memory operation to CUDA stream
            - [ ] send job-finish to job sender
            - [ ] a timely check
        - [ ] test guaranteed properties
            - [ ] first-in-first-done for received jobs
    - [ ] an actor API for multiple CUDA streams locally
        - [ ] ...
        - [ ] implement data-split and distribution to sub-worker actors
        - [ ] test guaranteed properties
            - [ ] first-in-first-done property for received jobs
    - [ ] an actor API for multiple CUDA streams on one net-positioned machine
        - [ ] ...
        - [ ] authentication
    - [ ] an actor API for distributed CUDA streams
        - [ ] ...
- [ ] tensor operation layer
    - [ ] actor API for tensor
        - [ ] implement internal state
            - [ ] data segments
            - [ ] size info
            - [ ] forward counter
            - [ ] backward hook stack (functions with internal states on data and grad)
        - [ ] implement message-receive
            - [ ] counter increment -> increase forward counter
            - [ ] counter decrement -> decrease forward counter, if zero, launch backward hooks
            - [ ] add hook
            - [ ] initialize from local data
        - [ ] implement message-send
            - [ ] forward counter increase done
    - [ ] user-friendly API for tensor (utilize actor API)
        - [ ] implement fancy indexing like `ndarray` package in python
        - [ ] implement Einstein summation convention with backward
        - [ ] implement max and min with backward
        - [ ] implement reshape with backward
        - [ ] implement `(+) (-) (*) (/)`
        - [ ] implement self-consuming version for these above operations
        - [ ] implement convolution
    - [ ] proc-macros for compilation level optimization
        - [ ] graph IR optimization (like apache TVM)
            - [ ] ... (learning about compilers)

- [ ] neural network layers (implemented utilizing the above things)
  - [ ] tensor-in-tensor-out
    - [ ] ...
  - [ ] data-in-tensor-out (tokenizer stuff)
    - [ ] ...

