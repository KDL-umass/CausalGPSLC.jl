Contributing to GPSLC.jl
=====

# Development

For basic editing of small portions of the GPSLC.jl source code, 
you can use `Revise.jl` to ensure the package is precompiled and up-to-date.

## Forks

If you intend to make a contribution to `GPSLC.jl`, 
please make a fork of this repository
and work on your additions there.

## Pull Requests

When the changes on your fork are ready for review, 
please create a pull request on Github to the `dev` branch of `GPSLC.jl`.
Please also mention any relevant issues the PR is intending to address in the 
description of the PR.

If your changes aren't ready for submission but you would like feedback, 
please create a pull request on Github as above, but mark it as a `Draft`.
This will let reviewers know that the PR is in-progress and looking for feedback.
Feel free to add any extra comments or extend the description to indicate
what you would like reviwed.

## Revise

In the julia REPL execute

```julia
using Pkg, Revise
Pkg.activate(".")
using GPSLC
```

And then you're good to go and add things as needed and rerun them. Revise should keep things up to date in the REPL as changes are made.

## Testing

All changes should be accompanied by corresponding tests in the test suite found in `test/runtests.jl`. 
We recommend using test-driven-development (TDD) and writing tests for the components prior to the components themselves, so you know they work as expected once they've been written. 
There are many good resources out there for TDD.

If you're using Visual Studio Code with the Julia extension, the contents of `runtests.jl`, the `GPSLCTests` module can be executed using `Shift+Enter` or similar, 
and it will include a call to `Revise.jl` so that the `GPSLC.jl` package is up to date.

You can also run the tests straight from the REPL by running

```julia
include("test/runtests.jl")
```

or to run the full test suite including package installation verification

```julia
import Pkg; Pkg.test("GPSLC")
```