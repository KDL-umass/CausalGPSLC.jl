# GPSLC: Gaussian Processes with Structured Latent Confounders

[![Build Status](https://github.com/jackkenney/GPSLC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jackkenney/GPSLC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/jackkenney/GPSLC.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/jackkenney/GPSLC.jl)


## Description

This code provides a working example of the algorithm 3 in the ICML 2020 [paper](http://proceedings.mlr.press/v119/witty20a/witty20a.pdf). In summary, this code estimates posterior distributions over individual treatment effects given an observational dataset and an intervention assignment.

## Examples

Running the examples can be done with

```bash
julia examples/basicExample.jl
```

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
