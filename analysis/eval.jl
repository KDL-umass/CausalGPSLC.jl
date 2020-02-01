using Gen
using LinearAlgebra
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
include("../data/processing_IHDP.jl")
include("../src/estimation.jl")
include("../baseline/multilevel_model.jl")
include("../data/synthetic.jl")

using .Model
using .Inference
using .ProcessingISO
using .MultilevelModel
using .Estimation
using .ProcessingIHDP
using .Synthetic


# load ISO data and subsample to bias observation
function load_ISO(experiment)
    config_path = "../experiments/config/ISO/$(experiment).toml"
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
    importanceWeights = ProcessingISO.generateImportanceWeights(new_means, new_vars, weekday_df)
    T, Y, SigmaU, regions_key = ProcessingISO.resampleData(config["downsampling"]["nSamplesPerState"], importanceWeights, weekday_df)

    # Scale T and Y
    T /= 100
    Y /= 10000
    T, Y, regions_key, df
end


"""
Fit GPR with full data before biased sub-sampling
we use the predicted mean as "true" counterfactuals
"""
function true_Ycf_ISO(doTs::Vector{Float64}, Ts, Ys)
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


# load IHDP data with true counterfactuals
function load_IHDP(experiment)
    config_path = "../experiments/config/IHDP/$(experiment).toml"
    config = TOML.parsefile(config_path)
    Random.seed!(config["data_params"]["seed"])

    data = DataFrame(CSV.File(config["paths"]["data"]))[1:config["data_params"]["nData"], :]
    pairs = ProcessingIHDP.generatePairs(data, config["data_params"]["pPair"])
    nData = size(data)[1]
    n = nData + length(pairs)
    SigmaU = ProcessingIHDP.generateSigmaU(pairs, nData)

    SigmaU = ProcessingIHDP.generateSigmaU(pairs, nData)
    T_ = ProcessingIHDP.generateT(data, pairs)
    T = [Float64(t) for t in T_]
    doTs = [Float64(1-t) for t in T_]

    X_ = Array(ProcessingIHDP.generateX(data, pairs))

    U = ProcessingIHDP.generateU(data, pairs)
    BetaX, BetaU = ProcessingIHDP.generateWeights(config["data_params"]["weights"], config["data_params"]["weightsP"])
    Y_, Ycf = ProcessingIHDP.generateOutcomes(X_, U, T_, BetaX, BetaU, config["data_params"]["CATT"], n)
    Y = [Float64(y) for y in Y_]
    obj_key = vcat([i for i in 1:200], pairs)

    Ycfs = Dict() # obj -> doT -> list
    for (i, obj) in enumerate(Set(obj_key))
        Ycfs[obj] = Dict()
        for doT in Set(doTs)
            Ycfs[obj][doT] = []
        end
    end
    for (i, obj) in enumerate(obj_key)
        push!(Ycfs[obj][doTs[i]], Ycf[i])
    end
    doTs = [0.0, 1.0]
    return T, doTs, X_, Y, Ycfs, obj_key
end


# load synthetic data with true counterfactuals
function load_synthetic(experiment)
    config_path = "../experiments/config/synthetic/$(experiment).toml"
    config = TOML.parsefile(config_path)

    data_config_path = config["paths"]["data"]
    SigmaU, U_, T_, X_, Y_, epsY, ftxu = generate_synthetic_confounder(data_config_path)
    nX = size(X_)[2]
    n = length(T_)

    obj_size = TOML.parsefile(data_config_path)["data"]["obj_size"]
    label = 1
    obj_label = zeros(n)
    for i in 1:n
        obj_label[i] = label
        if (i % obj_size) == 0
            label += 1
        end
    end
    obj_key = Int.(obj_label)

    if maximum(T_) == 1.0
        T = [t for t in T_]
        doTs = [0.0, 1.0]
        binary = true
    else
        T = T_
        doTnSteps = 20
        lower = minimum(T) + 0.05 * (maximum(T) - minimum(T))
        upper = maximum(T) - 0.05 * (maximum(T) - minimum(T))

        doTstepSize = (upper - lower)/doTnSteps

        doTs = [doT for doT in lower:doTstepSize:upper]
        binary = false
    end

    Ycfs = Dict() # obj -> doT -> list
    for (i, obj) in enumerate(Set(obj_key))
        Ycfs[obj] = Dict()
        for doT in Set(doTs)
            indeces = (obj_key .== obj) .& (T_ .!= doT)
            Ycfs[obj][doT] = ftxu(fill(doT, sum(indeces)), X_[indeces, :], U_[indeces, :], epsY[indeces])
        end
    end
    return T_, doTs, X_, Y_, Ycfs, obj_key
end


"""
Load data and return T, doTs, X, Y, Ycfs, obj_key
T : observed treatment
doTs : list of interventions
X : observed confounding
Y : observed outcome
Ycf : true counterfactuals
obj_key : list of keys (can be any type) that represents the object e.g. ["obj1", "obj1", "obj2" ...]
"""
function load_data(dataset, experiment)
    T, doTs, Y, obj_key, X, Ycfs = nothing, nothing, nothing, nothing, nothing, nothing
    if dataset == "ISO"
        doTs = [doT for doT in 0.25:0.01:0.75]
        T, Y, obj_key, df = load_ISO(experiment)
        weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]
        Ts = Dict()
        Ys = Dict()
        for region in Set(obj_key)
            Ts[region] = weekday_df[weekday_df[!, :State] .== region, :DryBulbTemp]/100
            Ys[region] = weekday_df[weekday_df[!, :State] .== region, :RealTimeDemand]/10000
        end
        Ycfs = true_Ycf_ISO(doTs, Ts, Ys)
    elseif dataset == "IHDP"
        T, doTs, X, Y, Ycfs, obj_key = load_IHDP(experiment)
    elseif dataset == "synthetic"
        T, doTs, X, Y, Ycfs, obj_key = load_synthetic(experiment)
    end
    T, doTs, X, Y, Ycfs, obj_key
end


"""
evaluate model given T, doTs, X, Y, Ycfs, obj_key
returns PEHE and Log likelihood per object (in Dict)
"""

function eval_model(posterior_dir::String, model::String, T::Vector{Float64}, doTs::Vector{Float64},
                    Y::Vector{Float64}, Ycfs, obj_key, nSamplesPerPost::Int,
                    nOuter::Int, burnIn::Int, stepSize::Int, stats_by_doT::Bool, bias)
    # convert obj_key to Int index
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

    # get posteriors for MLMs
    if model == "MLM_offset"
        posteriors = posteriorLinearMLMoffset(nOuter, T, Y, obj_label)
    elseif model == "MLM"
        posteriors = posteriorLinearMLM(nOuter, T, Y, obj_label)
    else
        posteriors = nothing
    end

    # data structure
    estIntLogLikelihoods = Dict() # obj -> doT
    estMeans = Dict() # obj -> doT -> list
    indecesDict = Dict()

    samples = Dict()
    for obj in objects
        indecesDict[obj] = obj_key .== obj
        estIntLogLikelihoods[obj] = Dict()
        estMeans[obj] = Dict()
        samples[obj] = Dict()
        for doT in doTs
            estIntLogLikelihoods[obj][doT] = []
            estMeans[obj][doT] = []
            samples[obj][doT] = []
        end
    end

    # get results from each posterior sample
<<<<<<< HEAD
    if model != "BART"
        for i in tqdm(burnIn:stepSize:nOuter)
            if occursin("MLM", model)
                post = posteriors[i]
=======
    for i in tqdm(burnIn:stepSize:nOuter)
        if occursin("MLM", model)
            post = posteriors[i]
        elseif model == "GP_per_object"
            posts = Dict()
            for obj in objects
            	posts[obj] = load("../experiments/" * posterior_dir * "/$(obj)Posterior$(i).jld")
            end
        else
            post = load("../experiments/" * posterior_dir * "/Posterior$(i).jld")
            if model == "no_confounding"
                uyLS = nothing
                U = nothing
            else
                uyLS = convert(Array{Float64,1}, post["uyLS"])
                U = post["U"]
            end
        end
        estMean = []
        for (j, doT) in enumerate(doTs)
            if model == "MLM_offset"
                MeanITE, CovITE = predictionMLMoffset(post, doT, obj_label)
            elseif model == "MLM"
                MeanITE, CovITE = predictionMLM(post, doT, obj_label)
>>>>>>> 02c01934408b9367c70fc171775437ea3f19af6a
            elseif model == "GP_per_object"
                post = nothing
            else
<<<<<<< HEAD
                post = load("../experiments/" * posterior_dir * "/Posterior$(i).jld")
                if model == "no_confounding"
                    uyLS = nothing
                    U = nothing
                else
                    uyLS = convert(Array{Float64,1}, post["uyLS"])
                    U = post["U"]
=======
                MeanITE, CovITE = conditionalITE(uyLS,
                                              post["tyLS"],
                                              nothing,
                                              post["yNoise"],
                                              post["yScale"],
                                              U,
                                              nothing,
                                              T,
                                              Y,
                                              doT)
            end
            for obj in objects
                indeces = indecesDict[obj]
                if model == "GP_per_object"
                    post = posts[obj]
                    MeanITE, CovITE = conditionalITE(nothing,
                                              post["tyLS"],
                                              nothing,
                                              post["yNoise"],
                                              post["yScale"],
                                              nothing,
                                              nothing,
                                              T[indeces],
                                              Y[indeces],
                                              doT)
>>>>>>> 02c01934408b9367c70fc171775437ea3f19af6a
                end
            end
            estMean = []
            for (j, doT) in enumerate(doTs)
                if model == "MLM_offset"
                    MeanITE, CovITE = predictionMLMoffset(post, doT, obj_label)
                elseif model == "MLM"
                    MeanITE, CovITE = predictionMLM(post, doT, obj_label)
                elseif model == "GP_per_object"
                    MeanITE, CovITE = nothing, nothing
                else
                    MeanITE, CovITE = conditionalITE(uyLS,
                                                  post["tyLS"],
                                                  nothing,
                                                  post["yNoise"],
                                                  post["yScale"],
                                                  U,
                                                  nothing,
                                                  T,
                                                  Y,
                                                  doT)
                end
                for obj in objects
                    indeces = indecesDict[obj]
                    if model == "GP_per_object"
                        post = load("../experiments/" * posterior_dir * "/$(obj)Posterior$(i).jld")
                        MeanITE, CovITE = conditionalITE(nothing,
                                                  post["tyLS"],
                                                  nothing,
                                                  post["yNoise"],
                                                  post["yScale"],
                                                  nothing,
                                                  nothing,
                                                  T[indeces],
                                                  Y[indeces],
                                                  doT)
                    end
                    if occursin("MLM", model)
                        m = mean(MeanITE[indeces])
                        v = mean(CovITE[indeces])
                    elseif model == "GP_per_object"
                        m = mean(MeanITE) + mean(Y[indeces])
                        v = mean(CovITE)
                    else
                        m = mean(MeanITE[indeces]) + mean(Y[indeces])
                        v = mean(CovITE[indeces, indeces])
                    end

                    # aggregate loglikelihood and errors
                    truth = Ycfs[obj][j]

                    estIntLogLikelihood = loglikelihood(Normal(m, v), [truth])

                    for sample in 1:nSamplesPerPost
                        push!(samples[obj][doT], normal(m, v))
                    end

                    push!(estIntLogLikelihoods[obj][doT], estIntLogLikelihood)

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

    else # BART
        Ycf_pred = load("BART_results/ISO/bias_$(bias).jld")
    end



    # calculate statistics. Important that this is shared
    errors = Dict()
    for obj in objects
        errors[obj] = (mean((Ycf_pred[obj] .- Ycfs[obj]).^2))^0.5
    end


    scores = Dict()
    logmeanexp(x) = logsumexp(x)-log(length(x))
    for obj in objects
        scores[obj] = 0
        for doT in doTs
            if model != "BART"
                scores[obj] += logmeanexp([Real(llh) for llh in estIntLogLikelihoods[obj][doT]])
            else
                scores[obj] = 0
            end
        end
        scores[obj] /= length(doTs)
    end


    errors, scores, samples

end


"""
evaluation with covariates
model evalutes CATE
"""
function eval_model(posterior_dir::String, model::String, T::Vector{Float64}, doTs::Vector{Float64}, X, Y::Vector{Float64}, Ycfs, obj_key,
    nSamplesPerPost::Int, nOuter::Int, burnIn::Int, stepSize::Int, stats_by_doT::Bool, bias)

    # convert obj_key to Int index
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

    # get posteriors for MLMs
    if model == "MLM_offset"
        posteriors = posteriorLinearMLMoffset(nOuter, T, X, Y, obj_label)
    elseif model == "MLM"
        posteriors = posteriorLinearMLM(nOuter, T, X, Y, obj_label)
    else
        posteriors = nothing
    end

    # data structure
    estIntLogLikelihoods = Dict() # obj -> doT
    estMeans = Dict() # obj -> doT -> list
    indecesDict = Dict()
    samples = Dict()

    for object in objects
        indecesDict[object] = Dict()
        estIntLogLikelihoods[object] = Dict()
        estMeans[object] = Dict()
        for doT in doTs
            estIntLogLikelihoods[object][doT] = []
            estMeans[object][doT] = []
            indecesDict[object][doT] = (obj_label .== object) .& (T .!= doT)
        end
    end
<<<<<<< HEAD
    if model != "BART"
        # get results from each posterior sample
        n, nX = size(X)
        X_lst = [X[:, i] for i in 1:nX]
        for i in tqdm(burnIn:stepSize:nOuter)
            if occursin("MLM", model)
                post = posteriors[i]
            elseif model == "GP_per_object"
                post = nothing
=======

    # get results from each posterior sample
    n, nX = size(X)
    X_lst = [X[:, i] for i in 1:nX]
    for i in tqdm(burnIn:stepSize:nOuter)
        if occursin("MLM", model)
            post = posteriors[i]
        elseif model == "GP_per_object"
            posts = Dict()
            for obj in objects
            	posts[obj] = load("../experiments/" * posterior_dir * "/$(obj)Posterior$(i).jld")
            end
        else
            post = load("../experiments/" * posterior_dir * "/Posterior$(i).jld")
            xyLS = convert(Array{Float64,1}, post["xyLS"])
            if model == "no_confounding"
                uyLS = nothing
                U = nothing
>>>>>>> 02c01934408b9367c70fc171775437ea3f19af6a
            else
                post = load("../experiments/" * posterior_dir * "/Posterior$(i).jld")
                xyLS = convert(Array{Float64,1}, post["xyLS"])
                if model == "no_confounding"
                    uyLS = nothing
                    U = nothing
                else
                    uyLS = convert(Array{Float64,1}, post["uyLS"])
                    U = post["U"]
                end
            end

            for (j, doT) in enumerate(doTs)

<<<<<<< HEAD
                if model == "MLM_offset"
                    MeanITE, CovITE = predictionMLMoffset(post, doT, X, obj_label)
                elseif model == "MLM"
                    MeanITE, CovITE = predictionMLM(post, doT, X, obj_label)
                elseif model == "GP_per_object"
                    MeanITE, CovITE = nothing, nothing
                else
                    MeanITE, CovITE = conditionalITE(uyLS,
                                                  post["tyLS"],
                                                  xyLS,
                                                  post["yNoise"],
                                                  post["yScale"],
                                                  U,
                                                  X_lst,
                                                  T,
                                                  Y,
                                                  doT)
=======
            if model == "MLM_offset"
                MeanITE, CovITE = predictionMLMoffset(post, doT, X, obj_label)
            elseif model == "MLM"
                MeanITE, CovITE = predictionMLM(post, doT, X, obj_label)
            elseif model == "GP_per_object"
                MeanITE, CovITE = nothing, nothing
            else
                MeanITE, CovITE = conditionalITE(uyLS,
                                              post["tyLS"],
                                              xyLS,
                                              post["yNoise"],
                                              post["yScale"],
                                              U,
                                              X_lst,
                                              T,
                                              Y,
                                              doT)
            end
            for obj in objects
                mask = indecesDict[obj][doT]
                if model == "GP_per_object"
                    post = posts[obj]
                    xyLS = convert(Array{Float64,1}, post["xyLS"])
                    MeanITE, CovITE = conditionalITE(nothing,
                                              post["tyLS"],
                                              xyLS,
                                              post["yNoise"],
                                              post["yScale"],
                                              nothing,
                                              [x[mask] for x in X_lst],
                                              T[mask],
                                              Y[mask],
                                              doT)
>>>>>>> 02c01934408b9367c70fc171775437ea3f19af6a
                end
                for obj in objects
                    mask = indecesDict[obj][doT]
                    if model == "GP_per_object"
                        post = load("../experiments/" * posterior_dir * "/Object$(obj)Posterior$(i).jld")
                        xyLS = convert(Array{Float64,1}, post["xyLS"])
                        MeanITE, CovITE = conditionalITE(nothing,
                                                  post["tyLS"],
                                                  xyLS,
                                                  post["yNoise"],
                                                  post["yScale"],
                                                  nothing,
                                                  [x[mask] for x in X_lst],
                                                  T[mask],
                                                  Y[mask],
                                                  doT)
                    end
                    if sum(mask) != 0
                        if occursin("MLM", model)
                            m = MeanITE[mask]
                            v = Diagonal(CovITE[mask])
                        elseif model == "GP_per_object"
                            m = MeanITE .+ Y[mask]
                            v = Symmetric(CovITE) + I*(1e-10)
                        else
                            m = MeanITE[mask] .+ Y[mask]
                            v = Symmetric(CovITE[mask, mask]) + I*(1e-10)
                        end
                        # aggregate loglikelihood and errors
                        truth = Ycfs[obj][doT]
                        truthLogLikelihood = Distributions.logpdf(MvNormal(m, v), truth) / length(truth)
                        push!(estIntLogLikelihoods[obj][doT], truthLogLikelihood)
                        push!(estMeans[obj][doT], m)
                    end
                end
            end
        end
    end
    errors, scores = Dict(), Dict()
    logmeanexp(x) = logsumexp(x)-log(length(x))
    if model != "BART"
        if stats_by_doT
            for doT in doTs
                scores[doT] = []
                for obj in objects
                    if length(estIntLogLikelihoods[obj][doT]) != 0
                        push!(scores[doT], logmeanexp([Real(llh) for llh in estIntLogLikelihoods[obj][doT]]))
                    end
                end
                scores[doT] = mean(scores[doT])
            end
        else
            for obj in objects
                scores[obj] = []
                for doT in doTs
                    if length(estIntLogLikelihoods[obj][doT]) != 0
                        push!(scores[obj], logmeanexp([Real(llh) for llh in estIntLogLikelihoods[obj][doT]]))
                    end
                end
                scores[obj] = mean(scores[obj])
            end
        end

        # average over posterior samples
        Ycf_pred = Dict()
        for obj in objects
            Ycf_pred[obj] = Dict()
            for (j, doT) in enumerate(doTs)
                Ycf_pred[obj][doT] = zeros(sum(indecesDict[obj][doT]))
                for n in 1:length(estMeans[obj][doT])
                    Ycf_pred[obj][doT] .+= (estMeans[obj][doT][n]./length(estMeans[obj][doT]))
                end
            end
        end
    else # BART
        Ycf_pred = load(bias)
    end

    # calculate statistics. Important that this is shared
    if stats_by_doT
        for (i, doT) in enumerate(sort(doTs))
            errors[doT] = []
            for obj in objects
                if model == "BART"
                    obj_b = string(obj)
                    doT_b = string(doT)
                else
                    obj_b = obj
                    doT_b = doT
                end
                if length(Ycf_pred[obj_b][doT_b]) != 0
                    push!(errors[doT], (mean((Ycf_pred[obj_b][doT_b] .- Ycfs[obj][doT]).^2))^0.5)
                end
            end
            # average over objects
            errors[doT] = mean(errors[doT])
        end
    else
        for obj in objects
            errors[obj] = []
            for (i, doT) in enumerate(sort(doTs))
                if model == "BART"
                    obj_b = string(obj)
                    doT_b = string(doT)
                else
                    obj_b = obj
                    doT_b = doT
                end
#                 println(bias, " ", doT, " ", doT_b)
                if length(Ycf_pred[obj_b][doT_b]) != 0
                    push!(errors[obj], (mean((Ycf_pred[obj_b][doT_b] .- Ycfs[obj][doT]).^2))^0.5)
                end
            end
            errors[obj] = mean(errors[obj])
        end
    end

    errors, scores, samples
end


"""
Main method. Takes a path to config file
"""
function main(args)

    experiment = args[1]
    baseline_model = args[2]
    dataset = args[3]
    nOuter = parse(Int64, args[4])
    burnIn = parse(Int64, args[5])
    stepSize = parse(Int64, args[6])
    nSamplesPerPost = parse(Int64, args[7])

    exp_config_path = "../experiments/config/$(dataset)/$(experiment).toml"
    exp_config = TOML.parsefile(exp_config_path)
    model = exp_config["model"]["type"]
    posterior_dir = exp_config["paths"]["posterior_dir"]

    if dataset == "ISO"
        bias = exp_config["downsampling"]["bias"]
<<<<<<< HEAD
    else
        data_config_path = exp_config["paths"]["data"]
        data_id = split(data_config_path, "/")[end]
        bias = split(data_id, ".")[1]
        bias = "BART_results/$(dataset)/$(bias).jld"
=======
        nSamplesPerState = exp_config["downsampling"]["nSamplesPerState"]
>>>>>>> 02c01934408b9367c70fc171775437ea3f19af6a
    end

    if baseline_model != "nothing"
        if model != "correct"
            # A terrible hack to not waste swarm2 runtime.
            return
        end
        model = baseline_model
    end

    # load evaluation data
    T, doTs, X, Y, Ycfs, obj_key = load_data(dataset, experiment)
    stats_by_doT = (dataset == "IHDP")
    if dataset == "ISO"
        errors, scores, samples = eval_model(posterior_dir, model, T, doTs, Y, Ycfs, obj_key, nSamplesPerPost, nOuter, burnIn, stepSize, stats_by_doT, bias)
    else
        println(data_id)
        errors, scores, samples = eval_model(posterior_dir, model, T, doTs, X, Y, Ycfs, obj_key, nSamplesPerPost, nOuter, burnIn, stepSize, stats_by_doT, bias)
    end

    # generate CSV
    df = Dict()
    df["model"] = []
    df["obj"] = []
    df["error"] = []
    if model != "BART"
        df["likelihood"] = []
    end

    if stats_by_doT
        label_keys = doTs
    else
        label_keys = obj_key
    end
    for obj in Set(label_keys)
        push!(df["model"], model)
        push!(df["obj"], obj)
        push!(df["error"], errors[obj])
        if model != "BART"
            push!(df["likelihood"], scores[obj])
        end
    end

    if dataset == "ISO"
    	if parse(Int64, experiment) < 41
        	save("results/$(dataset)/bias$(bias)/$(model)_samples.jld", samples)
        	CSV.write("results/$(dataset)/bias$(bias)/$(model)_scores.csv", DataFrame(df))
        else
        	save("results/$(dataset)/nSamplesPerState$(nSamplesPerState)/$(model)_samples.jld", samples)
        	CSV.write("results/$(dataset)/nSamplesPerState$(nSamplesPerState)/$(model)_scores.csv", DataFrame(df))
        end
    else
        data_config_path = exp_config["paths"]["data"]
        data_id = split(data_config_path, "/")[end]
        data_id = split(data_id, ".")[1]
        CSV.write("results/$(dataset)/$(experiment)_$(model)_scores.csv", DataFrame(df))
    end
end

main(ARGS)
