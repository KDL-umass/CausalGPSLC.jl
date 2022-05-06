GPSLC Tests
======

Most users will run `runtests.jl` without any arguments 
as this includes the rest and initializes.

Users who do not have an active REPL can run all the tests using:

```bash
julia -e 'import Pkg;Pkg.activate("."); include("test/runtests.jl")'      
```
