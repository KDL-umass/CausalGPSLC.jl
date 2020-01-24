# +
using Gen
using LinearAlgebra
using PyPlot
using TOML
using JLD
using CSV
using DataFrames
using Statistics
using Distributions

include("../src/model.jl")
include("../src/estimation.jl")
include("../src/inference.jl")

using .Model
using .Estimation
using .Inference
logmeanexp(x) = logsumexp(x)-log(length(x))

# +
# experiment = ARGS[1]
experiment = 4

config_path = "../experiments/config/ISO/$(experiment).toml"
config = TOML.parsefile(config_path)
println()
# -

# Load Fixed Data
T = load("../experiments/" * config["paths"]["posterior_dir"] * "T.jld")["T"]
Y = load("../experiments/" * config["paths"]["posterior_dir"] * "Y.jld")["Y"]
regions_key = load("../experiments/" * config["paths"]["posterior_dir"] * "regions_key.jld")["regions_key"]
println()

colors = ["black", "red", "orange", "blue", "brown", "grey"]
regions = Set(regions_key)

# +
# Ground Truth Estimate
df = DataFrame(CSV.File("../experiments/" * config["paths"]["data"]))
weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

Ts = Dict()
Ys = Dict()

for region in regions
    Ts[region] = weekday_df[weekday_df[!, :Region] .== region, :DryBulbTemp]/100
    Ys[region] = weekday_df[weekday_df[!, :Region] .== region, :RealTimeDemand]/1000
    scatter(Ts[region], Ys[region], label=region)
end
legend()
xlim(0.2, 0.8)
ylim(0, 4.2)
title("Original Data")
xlabel("T")
ylabel("Y")


# +
LS = 0.1
yNoise = 0.2
yScale = 1.
doTs = [doT for doT in 0.25:0.01:0.75]

truthIntMeans = Dict()
truthIntVars = Dict()

figure()

for (i, region) in enumerate(regions)
    kTT = processCov(rbfKernelLog(Ts[region], Ts[region], LS), yScale, yNoise)
    means = []
    vars = []
    for doT in doTs
        kTTs = processCov(rbfKernelLog(Ts[region], [doT], LS), yScale, nothing)
        kTsTs = processCov(rbfKernelLog([doT], [doT], LS), yScale, nothing)
        push!(means, (kTTs' * (kTT \ Ys[region]))[1])
        push!(vars, (kTsTs - kTTs' * (kTT \ kTTs))[1])
    end
    truthIntMeans[region] = means
    truthIntVars[region] = vars
    
    plot(doTs, means, color=colors[i])
    scatter(Ts[region], Ys[region], color=colors[i], label=region)
end
xlim(0.25, 0.75)
ylim(0, 4.2)
legend()
ylabel("E(Y|doT)")
xlabel("doT")
title("Smoothed Data - Ground Truth Causal Estimate")

save("../experiments/" * config["paths"]["posterior_dir"] * "doTs.jld", "doTs", doTs)  
save("../experiments/" * config["paths"]["posterior_dir"] * "Ycf.jld", "truthIntMeans", truthIntMeans)  

# +
for region in regions
    indeces = (regions_key .== region)
    scatter(T[indeces], Y[indeces], label=region)
end

legend()
title("Subsampled 'Observational' Data")
xlabel("T")
ylabel("Y")
xlim(0.2, 0.8)
ylim(0, 4.2)

# +
nOuter = 5000
burnIn = 1000
stepSize = 10
nSamplesPerPost = 100

estIntSamples = Dict()
estIntLogLikelihoods = Dict()

indecesDict = Dict()

for region in regions
    indecesDict[region] = regions_key .== region
    
    estIntSamples[region] = Dict()
    estIntLogLikelihoods[region] = Dict()
    
    for doT in doTs
        estIntSamples[region][doT] = []
        estIntLogLikelihoods[region][doT] = []
    end
end


if config["model"]["type"] == "GP_per_object"
    for i in burnIn:stepSize:nOuter
        for region in regions
            post = load("../experiments/" * config["paths"]["posterior_dir"] * "$(region)Posterior$(i).jld")
            indeces = indecesDict[region]
            
            for (j, doT) in enumerate(doTs)
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
                
                mean = sum(MeanITE)/sum(indeces) + sum(Y[indeces])/sum(indeces)
                var = sum(CovITE)/sum(indeces)^2
                
                truth = truthIntMeans[region][j]
                truthLogLikelihood = loglikelihood(Normal(mean, var), [truth])
                push!(estIntLogLikelihoods[region][doT], truthLogLikelihood)
                
                for _ in 1:nSamplesPerPost
                    sample = normal(mean, var)
                    push!(estIntSamples[region][doT], sample)
                end
            end
        end
    end
else
    for i in burnIn:stepSize:nOuter
        post = load("../experiments/" * config["paths"]["posterior_dir"] * "Posterior$(i).jld")

        if config["model"]["type"] == "no_confounding"
            uyLS = nothing
            U = nothing
        else
            uyLS = convert(Array{Float64,1}, post["uyLS"])
            U = post["U"]
        end
        
        for (j, doT) in enumerate(doTs)
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
            for region in regions
                indeces = indecesDict[region]
                mean = sum(MeanITE[indeces])/sum(indeces) + sum(Y[indeces])/length(Y[indeces])
                var = sum(CovITE[indeces, indeces])/sum(indeces)^2
                
                truth = truthIntMeans[region][j]
                truthLogLikelihood = loglikelihood(Normal(mean, var), [truth])
                push!(estIntLogLikelihoods[region][doT], truthLogLikelihood)  
                
                for _ in 1:nSamplesPerPost
                    sample = normal(mean, var)
                    push!(estIntSamples[region][doT], sample)
                end
            end
        end
    end
end
# + {}
# Compute percentile bounds

estIntMean = Dict()
estIntLower = Dict()
estIntUpper = Dict()

lower_bound = 0.025
upper_bound = 0.975

for region in regions
    estIntMean[region] = []
    estIntLower[region] = []
    estIntUpper[region] = []
    for doT in doTs
        sample = estIntSamples[region][doT]
        push!(estIntMean[region], mean(sample))
        push!(estIntLower[region], quantile(sample, lower_bound))
        push!(estIntUpper[region], quantile(sample, upper_bound))
    end
end

# +
# Plot Results
linewidth = 1

colors = ["black", "red", "orange", "blue", "brown", "grey"]

figure(figsize=(10,10))

for (i, region) in enumerate(regions)
    
    color = colors[i]
    plot(doTs*100, estIntMean[region]*1000, color = color, label=region)
    plot(doTs*100, estIntUpper[region]*1000, color = color, linestyle="--")
    plot(doTs*100, estIntLower[region]*1000, color = color, linestyle="--")
    scatter(Ts[region]*100, Ys[region]*1000, color=color, alpha=0.1)
    indeces = regions_key .== region
    scatter(T[indeces]*100, Y[indeces]*1000, color=color, marker="o")
end

legend()
xlim(minimum(doTs)*100, maximum(doTs)*100)
ylim(0, 4300)
xlabel("Temperature (F)")
ylabel("Average Daily Demand (MW)")
savefig("ISO_results/$(experiment).png")
# + {}
# Compute Numerical Evaluation
# http://www.stat.columbia.edu/~gelman/research/published/loo_stan.pdf

scores = Dict()

for region in regions
    scores[region] = 0
    
    for doT in doTs
        scores[region] += logmeanexp([Real(llh) for llh in estIntLogLikelihoods[region][doT]])
    end
    
    scores[region] /= length(doTs)
end


save("ISO_results/scores$(experiment).jld", scores)

# +

for region in regions
    println(region)
    println(load("ISO_results/scores1.jld")[region])
    println(load("ISO_results/scores2.jld")[region])
    println(load("ISO_results/scores3.jld")[region])
    println(load("ISO_results/scores4.jld")[region])
end
# -



