using Pkg
Pkg.activate(".")

using Random
Random.seed!(1234)

using GPSLC

function main()
    # load and prepare data
    X, T, Y, SigmaU = prepareData("examples/data/NEEC_sampled.csv")

    # do inference on latent values and the parameters
    println("Running Inference on U and Kernel Hyperparameters")
    ITEsamples = sampleITE(X, T, Y, SigmaU)

    # summarize results
    summarizeITE(ITEsamples; savetofile = "examples/results/NEEC_sampled_80.csv")
end

main()
