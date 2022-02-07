# Examples

These examples demonstrate the capabilities of GPSLC via a commandline tool submitted with the original ICML 2020 paper.

## Setup

1. Download and install julia 1.0 or later from https://julialang.org/downloads/
2. Execute the command `julia ` to install the necessary dependencies.
3. Execute the command `julia examples/estimateITE.jl` from the root of this repository to run the inference algorithm and save summary statistics of the estimated individual treatment effects including the mean and 90 percent credible intervals.

## Usage

By default, `julia examples/estimateITE.jl` uses the biased New England Energy Consumption data found in `data/NEEC_samples.csv` and an intervention assignment of 80 degrees Fahrenheit, saving the results in "results/NEEC_sampled_80.csv". The dataset, output filepath, intervention assignment, and inference hyperparameters can all be specified as command line arguments. For example,   

```bash
julia examples/estimateITE.jl --doT 0.0 --output_filepath examples/results/NEEC_samples_0.csv
```

runs the inference with an intervention assignment of 0 degrees Farenheit. For a full set of available command line arguments, run `julia examples/estimateITE.jl --help` in the command line. Note: The default number of inference steps is lower than shown in the paper results.

To run this script on external data, please follow the format in any of the sample data files in the `data` folder. Specifically, the csv file should include a column labeled "T", a column labeled "Y", any number of covariates, followed by a column labeled "obj". This implementation assumes that the instances are grouped together in the input data csv file.
