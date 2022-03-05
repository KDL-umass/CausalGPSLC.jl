using Pkg
Pkg.activate(".")

using ArgParse
using Random
Random.seed!(1234)

using GPSLC

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--datapath"
        help = "a path to the data"
        default = "examples/data/NEEC_sampled.csv"
        arg_type = String

        "--output_filepath"
        help = "filepath for inference results"
        default = "examples/results/NEEC_sampled_80.csv"
        arg_type = String

        "--doT"
        help = "treatment value to intervene"
        default = 0.8
        arg_type = Float64

        # posterior updates
        "--nOuter"
        help = "the number of posterior steps"
        default = 25
        arg_type = Int
        "--nMHInner"
        help = "the number of metropolis hastings sampling steps"
        default = 1
        arg_type = Int
        "--nESInner"
        help = "the number of elliptical slice sampling steps"
        default = 1
        arg_type = Int
        "--nU"
        help = "the dimension of latent confounders to model"
        default = 1
        arg_type = Int

        # inference
        "--burnIn"
        help = "the number of posterior samples for burn-in"
        default = 10
        arg_type = Int
        "--stepSize"
        help = "the step size during the inference step"
        default = 1
        arg_type = Int
        "--samplesPerPost"
        help = "the number of samples from each posterior for treatment effect approximation"
        default = 10
        arg_type = Int

        # parameters for priors
        "--uNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of U"
        default = 4.0
        arg_type = Float64
        "--uNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of U"
        default = 4.0
        arg_type = Float64
        "--xNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of X"
        default = 4.0
        arg_type = Float64
        "--xNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of X"
        default = 4.0
        arg_type = Float64
        "--tNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of T"
        default = 4.0
        arg_type = Float64
        "--tNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of T"
        default = 4.0
        arg_type = Float64
        "--yNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of Y"
        default = 4.0
        arg_type = Float64
        "--yNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of Y"
        default = 4.0
        arg_type = Float64
        "--xScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of X"
        default = 4.0
        arg_type = Float64
        "--xScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of X"
        default = 4.0
        arg_type = Float64
        "--tScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of T"
        default = 4.0
        arg_type = Float64
        "--tScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of T"
        default = 4.0
        arg_type = Float64
        "--yScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of Y"
        default = 4.0
        arg_type = Float64
        "--yScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of Y"
        default = 4.0
        arg_type = Float64
        "--uxLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and X"
        default = 4.0
        arg_type = Float64
        "--uxLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and X"
        default = 4.0
        arg_type = Float64
        "--utLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and T"
        default = 4.0
        arg_type = Float64
        "--utLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and T"
        default = 4.0
        arg_type = Float64
        "--xtLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of X and T"
        default = 4.0
        arg_type = Float64
        "--xtLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of X and T"
        default = 4.0
        arg_type = Float64
        "--uyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and Y"
        default = 4.0
        arg_type = Float64
        "--uyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and Y"
        default = 4.0
        arg_type = Float64
        "--xyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of X and Y"
        default = 4.0
        arg_type = Float64
        "--xyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of X and Y"
        default = 4.0
        arg_type = Float64
        "--tyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of T and Y"
        default = 4.0
        arg_type = Float64
        "--tyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of T and Y"
        default = 4.0
        arg_type = Float64
    end
    return parse_args(s)
end


function main()
    # parse arguments
    println("Parsing Arguments")
    parsed_args = parse_commandline()

    # load and prepare data
    X, T, Y, SigmaU = prepareData(parsed_args["datapath"])
    parsed_args["SigmaU"] = SigmaU

    # running GPSLC
    nOuter = parsed_args["nOuter"]
    nMHInner = parsed_args["nMHInner"]
    nESInner = parsed_args["nESInner"]
    nU = parsed_args["nU"]

    # do inference on latent values and the parameters
    println("Running Inference on U and Kernel Hyperparameters")
    posteriorSample = samplePosterior(X, T, Y, SigmaU; hyperparams = parsed_args,
        nU = nU, nOuter = nOuter, nMHInner = nMHInner, nESInner = nESInner)

    # inference of treatment effects
    burnIn = parsed_args["burnIn"]
    stepSize = parsed_args["stepSize"]
    samplesPerPost = parsed_args["samplesPerPost"]
    doT = parsed_args["doT"]

    # estimate individual treatment effects
    println("Estimating ITE")
    ITEsamples = sampleITE(X, T, Y, SigmaU; posteriorSample = posteriorSample,
        doT = doT, nU = nU, nOuter = nOuter,
        burnIn = burnIn, stepSize = stepSize, samplesPerPost = samplesPerPost)

    # summarize results
    summarizeITE(ITEsamples; savetofile = parsed_args["output_filepath"])
end

main()
