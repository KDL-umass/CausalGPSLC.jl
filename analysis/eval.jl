using Gen
using LinearAlgebra
using PyPlot
using TOML
using JLD
using CSV
using DataFrames
using Statistics
using Distributions
using Random
using ProgressBars

Random.seed!(1234)
include("../src/model.jl")
include("../src/inference.jl")
include("../data/processing_iso.jl")
include("../src/estimation.jl")
include("../baseline/multilevel_model.jl")

using .Model
using .Inference
using .ProcessingISO
using .MultilevelModel
using .Estimation


function load_ISO()
    # load experiment file
    config_path = "../experiments/config/ISO/1.toml"
    config = TOML.parsefile(config_path)

    bias = config["downsampling"]["bias"]
    mean = config["downsampling"]["mean"]
    newVar = config["downsampling"]["newVar"]
    regions = ["CT", "MA", "ME", "NH", "RI", "VT"]

    new_means = Dict()
    new_means["CT"] = mean + 3 * bias
    new_means["MA"] = mean + 2 * bias
    new_means["ME"] = mean + bias
    new_means["NH"] = mean - bias
    new_means["RI"] = mean - 2 * bias
    new_means["VT"] = mean - 3 * bias

    new_vars = Dict()
    for state in regions
        new_vars[state] = newVar
    end

    # Load and process data
    df = DataFrame(CSV.File(config["paths"]["data"]))
    weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

    # importanceWeights = generateImportanceWeights(config["new_means"], config["new_vars"], weekday_df)
    importanceWeights = generateImportanceWeights(new_means, new_vars, weekday_df)
    T, Y, SigmaU, regions_key = resampleData(config["downsampling"]["nSamplesPerState"], importanceWeights, weekday_df)

    # Scale T and Y
    T /= 100
    Y /= 1000
    T, Y, regions_key, df
end


function true_kernel_Ycf(doTs, Ts, Ys)
    LS = 0.1
    yNoise = 0.2
    yScale = 1.
    truthIntMeans = Dict()
    for (i, region) in tqdm(enumerate(keys(Ts)))
        kTT = processCov(rbfKernelLog(Ts[region], Ts[region], LS), yScale, yNoise)
        means = []
        vars = []
        for doT in doTs
            kTTs = processCov(rbfKernelLog(Ts[region], [doT], LS), yScale, nothing)
            kTsTs = processCov(rbfKernelLog([doT], [doT], LS), yScale, nothing)
            push!(means, (kTTs' * (kTT \ Ys[region]))[1])
        end
        truthIntMeans[region] = means
    end
    truthIntMeans
end


function load_data(dataset::String)
    T, doTs, Y, obj_key, X, Ycfs = nothing, nothing, nothing, nothing, nothing, nothing
    if dataset == "ISO"
        doTs = [doT for doT in 0.25:0.01:0.75]
        T, Y, obj_key, df = load_ISO()
        weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]
        Ts = Dict()
        Ys = Dict()
        for region in Set(obj_key)
            Ts[region] = weekday_df[weekday_df[!, :State] .== region, :DryBulbTemp]/100
            Ys[region] = weekday_df[weekday_df[!, :State] .== region, :RealTimeDemand]/1000
        end
        Ycfs = true_kernel_Ycf(doTs, Ts, Ys)
    end
    T, doTs, X, Y, Ycfs, obj_key
end


function eval_model(config, model::String, T::Vector{Float64}, doTs::Vector{Float64}, Y::Vector{Float64}, Ycfs, obj_key)
    obj2id = Dict()
    init = 1
    for k in obj_key
        if !(k in keys(obj2id))
            obj2id[k] = init
            init += 1
        end
    end
    obj_label = [Int(obj2id[k]) for k in obj_key]
    objects = keys(obj2id)

    nOuter = config["nOuter"]
    burnIn = config["burnIn"]
    stepSize = config["stepSize"]

    if model == "MLM_offset"
        posteriors = posteriorLinearMLMoffset(nOuter, T, Y, obj_label)
    elseif model == "MLM"
        posteriors = posteriorLinearMLM(nOuter, T, Y, obj_label)
    end

    # data structure
    estIntLogLikelihoods = Dict() # obj -> doT
    estMeans = Dict() # obj -> doT -> list
    indecesDict = Dict()

    for object in objects
        indecesDict[object] = obj_key .== object
        estIntLogLikelihoods[object] = Dict()
        estMeans[object] = Dict()
        for doT in doTs
            estIntLogLikelihoods[object][doT] = []
            estMeans[object][doT] = []
        end
    end

    for i in tqdm(burnIn:stepSize:nOuter)
        if occursin("MLM", model)
            post = posteriors[i]
        end
        estMean = []
        for (j, doT) in enumerate(doTs)
            if model == "MLM_offset"
                MeanITE, CovITE = predictionMLMoffset(post, doT, obj_label)
            elseif model == "MLM"
                MeanITE, CovITE = predictionMLM(post, doT, obj_label)
            end
            for obj in objects
                indeces = indecesDict[obj]
                m = mean(MeanITE[indeces])
                v = mean(CovITE[indeces])
                truth = Ycfs[obj][j]
                truthLogLikelihood = loglikelihood(Normal(m, v), [truth])
                push!(estIntLogLikelihoods[obj][doT], truthLogLikelihood)
                append!(estMeans[obj][doT], m)
            end
        end
    end

    Ycf_pred = Dict()
    for obj in objects
        Ycf_pred[obj] = []
        for (j, doT) in enumerate(doTs)
            append!(Ycf_pred[obj], mean(estMeans[obj][doT]))
        end
    end

    errors = Dict()
    for obj in objects
        errors[obj] = (mean((Ycf_pred[obj] .- Ycfs[obj]).^2))^0.5
    end

    scores = Dict()
    logmeanexp(x) = logsumexp(x)-log(length(x))
    for obj in objects
        scores[obj] = 0
        for doT in doTs
            scores[obj] += logmeanexp([Real(llh) for llh in estIntLogLikelihoods[obj][doT]])
        end

        scores[obj] /= length(doTs)
    end

    errors, scores
end

function main(args)

    config_path = args[1]
    config = TOML.parsefile(config_path)
    dataset = config["dataset"]
    # load evaluation data
    T, doTs, X, Y, Ycfs, obj_key = load_data(dataset)

    models = config["models"]
    model_errors, model_scores = Dict(), Dict()
    for m in models
        if dataset == "ISO"
            errors, scores = eval_model(config, m, T, doTs, Y, Ycfs, obj_key)
            model_errors[m] = errors
            model_scores[m] = scores
        end
    end

    save("$(dataset)_results/model_scores.jld", model_scores)
    save("$(dataset)_results/model_errors.jld", model_errors)

    # print errors
    for m in models
        errors = model_errors[m]
        scores = model_scores[m]
        println(m)
        for obj in Set(obj_key)
            println(obj)
            println(errors[obj], " ", scores[obj])
        end
        println()
    end
end

main(ARGS)