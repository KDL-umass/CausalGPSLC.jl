# +
using JLD
using PyPlot
using Statistics
using TOML
using CSV
using DataFrames

include("../data/processing_iso.jl")

using .ProcessingISO

# +
SMALL_SIZE = 16
MEDIUM_SIZE = 22

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

rcParams["font.size"] = MEDIUM_SIZE         # controls default text sizes
rcParams["axes.titlesize"] = MEDIUM_SIZE     # fontsize of the axes title
# rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

# rc('text', usetex=True)
rcParams["text.usetex"] = true

# +
bias = 9

model_samples = load("results/ISO/model_samples_$(bias).jld")
config_path = "../experiments/config/ISO/$(bias).toml"
config = TOML.parsefile(config_path)


# +
models = ["correct", "no_confounding", "no_objects", "GP_per_object", "MLM", "MLM_offset"]

model_key = Dict()
model_key["correct"] = "GP-SLC"
model_key["no_confounding"] = "NoConf"
model_key["no_objects"] = "NoObj"
model_key["GP_per_object"] = "GPperObj"
model_key["MLM"] = "MLM 1"
model_key["MLM_offset"] = "MLM 2"

states = ["CT", "MA", "ME", "NH", "RI", "VT"]
doTs   = sort([doT for doT in keys(model_samples["correct"]["CT"])])

println()

# +
df = DataFrame(CSV.File(config["paths"]["data"]))

weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

allTs = Dict()
allYs = Dict()

for state in states
    in_state = weekday_df[!, :State] .== state
    allTs[state] = weekday_df[in_state, :][!, :DryBulbTemp]
    allYs[state] = weekday_df[in_state, :][!, :RealTimeDemand]
end
# -



T = load("../experiments/results/ISO/bias$(bias)/correct/T.jld")["T"]
Y = load("../experiments/results/ISO/bias$(bias)/correct/Y.jld")["Y"]
regions_key = load("../experiments/results/ISO/bias$(bias)/correct/regions_key.jld")["regions_key"]
println()

scatter_color = "blue"
estimate_color = "green"

# +
# Get Mean, Lower Bound, and Upper Bounds

estIntMean = Dict()
estIntLower = Dict()
estIntUpper = Dict()

nModels = length(keys(model_samples))
nStates = length(keys(model_samples["correct"]))

lower_bound = 0.005
upper_bound = 0.995

fig, axes = subplots(nStates, nModels, figsize=(10,10), constrained_layout=true)
xlabel("Temperature")

samples = 0

for (i, model) in enumerate(models)
    estIntMean[model] = Dict()
    estIntLower[model] = Dict()
    estIntUpper[model] = Dict()
    for (j, state) in enumerate(states)
        estIntMean[model][state] = []
        estIntLower[model][state] = []
        estIntUpper[model][state] = []
        
        for doT in doTs
            samples = model_samples[model][state][doT]
            push!(estIntMean[model][state], mean(samples))
            push!(estIntLower[model][state], quantile(samples, lower_bound))
            push!(estIntUpper[model][state], quantile(samples, upper_bound))
        end
        
        index = (j-1) * nModels + i
        subplot(nStates, nModels, index)

        fill_between(doTs * 100, 
                     estIntUpper[model][state] * 10000, 
                     estIntLower[model][state] * 10000, 
                     alpha = 0.1,
                     color = estimate_color)
        plot(doTs * 100,
             estIntMean[model][state] * 10000,
             linestyle="--",
             color = estimate_color)
        in_state = regions_key .== state
        scatter(allTs[state], allYs[state], s=1, color=scatter_color, alpha=0.1)
        scatter(T[in_state] * 100, Y[in_state] * 10000, s=1, c=scatter_color, marker="o")

        xlim(25, 75)
#         ylim(0, 100000)
        
        if i == 1
            ylabel(state)
        end
        
        if j == 1
            title(model_key[model])
        end
        
        xticks([])
        yticks([])
    end
end


tight_layout(pad=0.4, w_pad=0.2, h_pad=0.2)
savefig("../figures/NEED_multiples.png", dpi=200)
# -




