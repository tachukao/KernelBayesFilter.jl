# KernelBayesFilter

Julia implementation of Kernel Bayes' Rule


```julia
includet("scripts/run.jl")

# run toy Kernel Bayes' rule example
Run.toy()

# run filtering example with original / IW method
Run.filtering(:original)
Run.filtering(:iw)
```