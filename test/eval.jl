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
    weekday_df = df[df[!, :IsWeekday].=="TRUE", :]

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
    yScale = 1.0
    truthIntMeans = Dict()
    for (i, region) in tqdm(enumerate(keys(Ts)))
        kTT = processCov(rbfKernelLog(Ts[region], Ts[region], LS), yScale, yNoise)
        means = []
        vars = []
        for doT in doTs
            kTTs = processCov(rbfKernelLog(Ts[region], [doT], LS), yScale)
            kTsTs = processCov(rbfKernelLog([doT], [doT], LS), yScale)
            push!(means, (kTTs'*(kTT\Ys[region]))[1])
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
    doTs = [Float64(1 - t) for t in T_]

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
    n, nX = size(X_)

    obj_size = TOML.parsefile(data_config_path)["data"]["obj_size"]
    label = 1
    obj_label = zeros(n)
    for i in 1:n
        obj_label[i] = label
        if (i % obj_size) == 0
            label += 1
        end
    end
    obj_key = Int64.(obj_label)

    if maximum(T_) == 1.0
        T = [t for t in T_]
        doTs = [0.0, 1.0]
        binary = true
    else
        T = T_
        doTnSteps = 20
        lower = minimum(T) + 0.05 * (maximum(T) - minimum(T))
        upper = maximum(T) - 0.05 * (maximum(T) - minimum(T))

        doTstepSize = (upper - lower) / doTnSteps

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
        weekday_df = df[df[!, :IsWeekday].=="TRUE", :]
        Ts = Dict()
        Ys = Dict()
        for region in Set(obj_key)
            Ts[region] = weekday_df[weekday_df[!, :State].==region, :DryBulbTemp] / 100
            Ys[region] = weekday_df[weekday_df[!, :State].==region, :RealTimeDemand] / 10000
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

function eval_model(posterior_dirs, model::String, T::Vector{Float64}, doTs::Vector{Float64},
    Y::Vector{Float64}, Ycfs, obj_key, nSamplesPerPost::Int64,
    nOuter::Int64, burnIn::Int64, stepSize::Int64, stats_by_doT::Bool, bart_pred)
    # convert obj_key to Int64 index
    obj2id = Dict()
    init = 1
    for k in obj_key
        if !(k in keys(obj2id))
            obj2id[k] = init
            init += 1
        end
    end
    obj_label = [Int64(obj2id[k]) for k in obj_key]
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
    if model != "BART"
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
                elseif model == "GP_per_object"
                    MeanITE, CovITE = nothing, nothing
                else

                    MeanITE, CovITE = conditionalITE(uyLS,
                        nothing,
                        post["tyLS"],
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
                            nothing,
                            post["tyLS"],
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
        Ycf_pred = bart_pred
    end

    # calculate statistics. Important that this is shared
    ITE_PEHE = 0
    for obj in objects
        ITE_PEHE += mean((Ycf_pred[obj] .- Ycfs[obj]) .^ 2) / length(objects)
    end
    errors = Dict()
    for obj in objects
        errors[obj] = ITE_PEHE
    end

    # accumulate over doT
    sate_true = Dict()
    sate_pred = Dict()
    SATE = 0
    for (i, doT) in enumerate(sort(doTs))
        sate_true[doT] = []
        sate_pred[doT] = []
        for obj in objects
            push!(sate_true[doT], Ycfs[obj][i])
            push!(sate_pred[doT], Ycf_pred[obj][i])
        end
        sate_true[doT] = mean(sate_true[doT])
        sate_pred[doT] = mean(sate_pred[doT]) # average over doT
        SATE += (mean(sate_true[doT]) - mean(sate_pred[doT]))^2 / length(doTs)
    end

    scores = Dict()
    for obj in objects
        scores[obj] = SATE
    end

    scores, errors, samples
end


"""
evaluation with covariates
model evalutes CATE
"""
function eval_model(posterior_dir, model::String, T::Vector{Float64}, doTs::Vector{Float64}, X, Y::Vector{Float64}, Ycfs, obj_key,
    nSamplesPerPost::Int64, nOuter::Int64, burnIn::Int64, stepSize::Int64, stats_by_doT::Bool, bart_pred)

    # convert obj_key to Int64 index
    obj2id = Dict()
    init = 1
    for k in obj_key
        if !(k in keys(obj2id))
            obj2id[k] = init
            init += 1
        end
    end
    obj_label = [Int64(obj2id[k]) for k in obj_key]
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

    if model != "BART"
        # get results from each posterior sample
        n, nX = size(X)
        X_lst = [X[:, i] for i in 1:nX]
        postislst = !isa(posterior_dir, String)
        if occursin("exp9", posterior_dir[9])
            posterior_dir[9] = posterior_dir[9] * "exp9"  # Is this an error?
        end
        for i in tqdm(burnIn:stepSize:nOuter)
            if occursin("MLM", model)
                post = posteriors[i]
            elseif model == "GP_per_object"
                posts = Dict()
                for obj in objects
                    if postislst
                        posts[obj] = [load("../experiments/" * p * "/$(obj)Posterior$(i).jld") for p in posterior_dir]
                    else
                        posts[obj] = load("../experiments/" * posterior_dir * "/Object$(obj)Posterior$(i).jld")
                    end
                end
            else
                if postislst

                    post = [load("../experiments/" * p * "Posterior$(i).jld") for p in posterior_dir]
                    xyLS = [convert(Array{Float64,1}, p["xyLS"]) for p in post]
                    if model == "no_confounding"
                        uyLS = nothing
                        U = nothing
                    else
                        uyLS = [convert(Array{Float64,1}, p["uyLS"]) for p in post]
                        U = [p["U"] for p in post]
                    end
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
            end

            for (j, doT) in enumerate(doTs)
                if model == "MLM_offset"
                    MeanITE, CovITE = predictionMLMoffset(post, doT, X, obj_label)
                elseif model == "MLM"
                    MeanITE, CovITE = predictionMLM(post, doT, X, obj_label)
                elseif model == "GP_per_object"
                    MeanITE, CovITE = nothing, nothing
                else
                    if postislst
                        MeanITE_agg = []
                        CovITE_agg = []

                        for (post_idx, p) in enumerate(post)
                            if uyLS == nothing
                                MeanITE, CovITE = conditionalITE(nothing,
                                    xyLS[post_idx],
                                    p["tyLS"],
                                    p["yNoise"],
                                    p["yScale"],
                                    nothing,
                                    X_lst,
                                    T,
                                    Y,
                                    doT)
                            else
                                MeanITE, CovITE = conditionalITE(uyLS[post_idx],
                                    xyLS[post_idx],
                                    p["tyLS"],
                                    p["yNoise"],
                                    p["yScale"],
                                    U[post_idx],
                                    X_lst,
                                    T,
                                    Y,
                                    doT)
                            end
                            push!(MeanITE_agg, MeanITE)
                            push!(CovITE_agg, CovITE)
                        end
                        MeanITE = mean(MeanITE_agg) # mean
                        CovITE = mean(CovITE_agg) # mean

                    else
                        MeanITE, CovITE = conditionalITE(uyLS,
                            xyLS,
                            post["tyLS"],
                            post["yNoise"],
                            post["yScale"],
                            U,
                            X_lst,
                            T,
                            Y,
                            doT)
                    end
                end

                for obj in objects
                    mask = indecesDict[obj][doT]
                    if model == "GP_per_object"
                        post = posts[obj]
                        if postislst
                            MeanITE_agg = []
                            CovITE_agg = []
                            xyLS = [convert(Array{Float64,1}, p["xyLS"]) for p in post]
                            for (i, p) in enumerate(post)
                                MeanITE, CovITE = conditionalITE(uyLS[i],
                                    xyLS[i],
                                    p["tyLS"],
                                    p["yNoise"],
                                    p["yScale"],
                                    U[i],
                                    X_lst,
                                    T,
                                    Y,
                                    doT)
                                push!(MeanITE_agg, MeanITE)
                                push!(CovITE_agg, CovITE)
                            end
                            MeanITE = mean(MeanITE_agg) # mean
                            CovITE = mean(CovITE_agg) # mean
                        else
                            xyLS = convert(Array{Float64,1}, post["xyLS"])
                            MeanITE, CovITE = conditionalITE(nothing,
                                xyLS,
                                post["tyLS"],
                                post["yNoise"],
                                post["yScale"],
                                nothing,
                                [x[mask] for x in X_lst],
                                T[mask],
                                Y[mask],
                                doT)
                        end
                    end
                    if sum(mask) != 0
                        if occursin("MLM", model)
                            m = MeanITE[mask]
                            v = Diagonal(CovITE[mask])
                        elseif model == "GP_per_object"
                            m = MeanITE .+ Y[mask]
                            v = Symmetric(CovITE) + I * (1e-10)
                        else
                            m = MeanITE[mask] .+ Y[mask]
                            v = Symmetric(CovITE[mask, mask]) + I * (1e-10)
                        end
                        push!(estMeans[obj][doT], m)
                    end
                end
            end
        end
    end
    error_ite, error_sate = Dict(), Dict()
    logmeanexp(x) = logsumexp(x) - log(length(x))
    if model != "BART"
        # average over posterior samples
        Ycf_pred = Dict()
        for obj in objects
            Ycf_pred[obj] = Dict()
            for (j, doT) in enumerate(doTs)
                Ycf_pred[obj][doT] = zeros(sum(indecesDict[obj][doT]))
                for n in 1:length(estMeans[obj][doT])
                    Ycf_pred[obj][doT] .+= (estMeans[obj][doT][n] ./ length(estMeans[obj][doT]))
                end
            end
        end
    else # BART
        Ycf_pred = bart_pred
    end

    # calculate statistics. Important that this is shared
    if stats_by_doT
        for (i, doT) in enumerate(sort(doTs))
            error_ite[doT] = []
            error_sate[doT] = []

            sate_true = []
            sate_pred = []
            for obj in objects
                if model == "BART"
                    obj_b = string(obj)
                    doT_b = string(doT)
                else
                    obj_b = obj
                    doT_b = doT
                end
                if length(Ycf_pred[obj_b][doT_b]) != 0
                    push!(error_ite[doT], mean((Ycf_pred[obj_b][doT_b] .- Ycfs[obj][doT]) .^ 2))
                    push!(sate_pred, mean(Ycf_pred[obj_b][doT_b])) # accumulate all predictions within object
                    push!(sate_true, mean(Ycfs[obj][doT])) # accumulate all ground-truth within object
                end
            end
            # average over objects
            error_ite[doT] = mean(error_ite[doT])
            # the difference of mean over object
            error_sate[doT] = (mean(sate_true) - mean(sate_pred))^2
        end
    else
        sate_true = Dict()
        sate_pred = Dict()
        for (i, doT) in enumerate(sort(doTs))
            sate_true[doT] = []
            sate_pred[doT] = []
        end

        for obj in objects
            error_ite[obj] = []
            for (i, doT) in enumerate(sort(doTs))
                if model == "BART"
                    obj_b = string(obj)
                    doT_b = string(doT)
                else
                    obj_b = obj
                    doT_b = doT
                end
                if length(Ycf_pred[obj_b][doT_b]) != 0
                    push!(error_ite[obj], mean((Ycf_pred[obj_b][doT_b] .- Ycfs[obj][doT]) .^ 2))
                    push!(sate_true[doT], mean(Ycfs[obj][doT])) # accumulate over objects
                    push!(sate_pred[doT], mean(Ycf_pred[obj_b][doT_b])) # accumulate over objects
                end
            end
        end

        SATE = 0
        for (i, doT) in enumerate(sort(doTs))
            # take the average over objects
            # average squared difference over doT
            SATE += ((mean(sate_true[doT]) - mean(sate_pred[doT]))^2) / length(doTs)
        end

        ITE_PEHE = 0
        for obj in objects
            ITE_PEHE += mean(error_ite[obj]) / length(objects)
        end

        for obj in objects
            error_sate[obj] = SATE
            error_ite[obj] = ITE_PEHE
        end

    end
    error_sate, error_ite, samples
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

    if dataset == "IHDP"
        experiment = parse(Int64, experiment)
        experiments = [i for i in experiment*10-9:experiment*10]
        posterior_dir = [TOML.parsefile("../experiments/config/IHDP/$(experiment).toml")["paths"]["posterior_dir"] for experiment in experiments] # list of posteriors
        model = TOML.parsefile("../experiments/config/IHDP/$(experiments[1]).toml")["model"]["type"]
    else
        posterior_dir = exp_config["paths"]["posterior_dir"]
    end

    if dataset == "ISO"
        bias = exp_config["downsampling"]["bias"]
        nSamplesPerState = exp_config["downsampling"]["nSamplesPerState"]
    else
        data_config_path = exp_config["paths"]["data"]
        data_id = split(data_config_path, "/")[end]
        data_id = split(data_id, ".")[1]
        bias = "BART_results/$(dataset)/$(data_id).jld"
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

    # load pretrained BART file
    bart_pred = nothing
    if model == "BART"
        if dataset == "ISO"
            if parse(Int64, experiment) < 41
                bart_pred = load("BART_results/ISO/bias_$(bias).jld")
            else
                bart_pred = load("BART_results/ISO/nSamplesPerState_$(nSamplesPerState).jld")
            end
        elseif dataset == "synthetic"
            bart_pred = load("BART_results/synthetic/$(data_id).jld")
        elseif dataset == "IHDP"
            bart_pred = load("BART_results/IHDP/1.jld")
        end
    end

    if dataset == "ISO"
        error_sate, error_ite, samples = eval_model(posterior_dir, model, T, doTs, Y, Ycfs, obj_key, nSamplesPerPost, nOuter, burnIn, stepSize, stats_by_doT, bart_pred)
    else
        error_sate, error_ite, samples = eval_model(posterior_dir, model, T, doTs, X, Y, Ycfs, obj_key, nSamplesPerPost, nOuter, burnIn, stepSize, stats_by_doT, bart_pred)
    end

    # generate CSV
    df = Dict()
    df["model"] = []
    df["obj"] = []
    df["error_sate"] = []
    df["error_ite"] = []
    if (model != "BART") & (dataset == "ISO")
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
        push!(df["error_sate"], error_sate[obj])
        push!(df["error_ite"], error_ite[obj])
    end

    if dataset == "ISO"
        if parse(Int64, experiment) < 41
            if model != "BART"
                save("results/$(dataset)/bias$(bias)/$(model)_samples.jld", samples)
            end
            CSV.write("results/$(dataset)/bias$(bias)/$(model)_scores.csv", DataFrame(df))
        else
            if model != "BART"
                save("results/$(dataset)/nSamplesPerState$(nSamplesPerState)/$(model)_samples.jld", samples)
            end
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
