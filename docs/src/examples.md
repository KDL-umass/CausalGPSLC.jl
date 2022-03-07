# Examples

## Simple example

The file `examples/basicExample.jl` has a simple use case for 
```@doc
julia> X, T, Y, SigmaU = prepareData("examples/data/NEEC_sampled.csv")

julia> println("Running Inference on U and Kernel Hyperparameters")
julia> posteriorSample = samplePosterior(X, T, Y, SigmaU)
julia> println("Estimating ITE")
julia> ITEsamples = sampleITE(X, T, Y, SigmaU; posteriorSample = posteriorSample)
julia> summarizeITE(ITEsamples; savetofile = "examples/results/NEEC_sampled_80.csv")
```


## Command Line Tool

These examples demonstrate the capabilities of GPSLC via a commandline tool submitted with the original ICML 2020 paper.

By default, `julia examples/estimateITE.jl` uses the biased New England Energy Consumption data found in `data/NEEC_samples.csv` and an intervention assignment of 80 degrees Fahrenheit, saving the results in "results/NEEC_sampled_80.csv". The dataset, output filepath, intervention assignment, and inference hyperparameters can all be specified as command line arguments. For example,   

```bash
julia examples/estimateITE.jl --doT 0.0 --output_filepath examples/results/NEEC_samples_0.csv
```

runs the inference with an intervention assignment of 0 degrees Farenheit. For a full set of available command line arguments, run `julia examples/estimateITE.jl --help` in the command line. Note: The default number of inference steps is lower than shown in the paper results.

To run this script on external data, please follow the format in any of the sample data files in the `data` folder. Specifically, the csv file should include a column labeled "T", a column labeled "Y", any number of covariates, followed by a column labeled "obj". This implementation assumes that the instances are grouped together in the input data csv file.

