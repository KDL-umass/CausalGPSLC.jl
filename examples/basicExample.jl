using Random
Random.seed!(1234)

using GPSLC

export basicExample

function basicExample(dataFile="examples/data/NEEC_sampled.csv")
    g = gpslc(dataFile)

    println("Estimating ITE")
    ITEsamples = sampleITE(g, doT=0.6)

    summarizeITE(ITEsamples; savetofile="examples/results/NEEC_sampled_0.6.csv")
end
