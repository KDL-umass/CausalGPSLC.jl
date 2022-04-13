# GPSLC: Gaussian Processes with Structured Latent Confounders

[![](https://img.shields.io/badge/language-julia-Green.svg)](https://julialang.org)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kdl-umass.github.io/GPSLC.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kdl-umass.github.io/GPSLC.jl/dev)
[![CI](https://github.com/KDL-umass/GPSLC.jl/workflows/CI/badge.svg)](https://github.com/kdl-umass/GPSLC.jl/actions?query=workflow%3ACI)
<!-- [![Codecov](https://codecov.io/gh/kdl-umass/GPSLC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kdl-umass/GPSLC.jl) -->

## Description

This code provides a working example of the algorithm 3 in the ICML 2020 [paper](http://proceedings.mlr.press/v119/witty20a/witty20a.pdf). In summary, this code estimates posterior distributions over individual treatment effects given an observational dataset and an intervention assignment.

## Examples

Running the examples can be done with

```bash
julia examples/basicExample.jl
```


## Contributing

Please review the contribution instructions in [CONTRIBUTING.md](CONTRIBUTING.md)

## Development

In the julia REPL execute

```julia
using Pkg, Revise
Pkg.activate(".")
using GPSLC
```

And then you're good to go and add things as needed and rerun them. Revise should keep things up to date in the REPL as changes are made.

## Acknowledgements

The original paper was published by Sam Witty, Kenta Takatsu, David Jensen, and Vikash Mansinghka in 2020.

This package was compiled by Jack Kenney in 2022 under the guidance of Sam Witty and David Jensen.
