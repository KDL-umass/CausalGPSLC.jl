using Random
Random.seed!(1234)

using GPSLC

export basicExample

function basicExample(dataFile = "examples/data/NEEC_sampled.csv")
    println("CURRENTLY IN: $(pwd())")
    X, T, Y, SigmaU = prepareData(dataFile)

    println("Running Inference on U and Kernel Hyperparameters")
    posteriorSample = samplePosterior(X, T, Y, SigmaU)

    println("Estimating ITE")
    ITEsamples = sampleITE(X, T, Y, SigmaU; posteriorSample = posteriorSample)

    summarizeITE(ITEsamples; savetofile = "examples/results/NEEC_sampled_80.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    basicExample()
end
