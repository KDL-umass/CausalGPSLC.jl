# +
using JLD
using PyPlot
using Statistics
using TOML
using CSV
using DataFrames


include("../data/processing_iso.jl")
include("../src/model.jl")

using .ProcessingISO
using .Model

# +
SMALL_SIZE = 16
MEDIUM_SIZE = 22

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")

rcParams["font.size"] = MEDIUM_SIZE         # controls default text sizes
rcParams["axes.titlesize"] = MEDIUM_SIZE     # fontsize of the axes title
rcParams["legend.fontsize"]= SMALL_SIZE    # legend fontsize
rcParams["legend.framealpha"] = 1.

# rc('text', usetex=True)
rcParams["text.usetex"] = true

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


println()

# +
bias = 9

model_samples = Dict()

for model in models
    model_samples[model] = load("results/ISO/bias$(bias)/$(model)_samples.jld")
end
    
doTs   = sort([doT for doT in keys(model_samples["correct"]["CT"])])
config_path = "../experiments/config/ISO/$(bias).toml"
config = TOML.parsefile(config_path)


# +
df = DataFrame(CSV.File(config["paths"]["data"]))

weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

allTs = Dict()
allYs = Dict()

for state in states
    in_state = weekday_df[!, :State] .== state
    allTs[state] = weekday_df[in_state, :][!, :DryBulbTemp]/100
    allYs[state] = weekday_df[in_state, :][!, :RealTimeDemand]/10000
end
# -
function true_Ycf_ISO(doTs::Vector{Float64}, Ts, Ys, LS, yNoise, yScale)
    truthIntMeans = Dict()
    for (i, region) in enumerate(keys(Ts))
        kTT = processCov(rbfKernelLog(Ts[region], Ts[region], LS), yScale, yNoise)
        means = []
        vars = []
        for doT in doTs
            kTTs = processCov(rbfKernelLog(Ts[region], [doT], LS), yScale, nothing)
            kTsTs = processCov(rbfKernelLog([doT], [doT], LS), yScale, nothing)
            push!(means, (kTTs' * (kTT \ Ys[region]))[1])
        end
        truthIntMeans[region] = means
        scatter(Ts[region], Ys[region], color="black")
        plot(doTs, means, color="red")
    end
    truthIntMeans
end


truthIntMeans = true_Ycf_ISO(doTs, allTs, allYs, 0.2, 0.3, 1.)

T = load("../experiments/results/ISO/bias$(bias)/correct/T.jld")["T"]
Y = load("../experiments/results/ISO/bias$(bias)/correct/Y.jld")["Y"]
regions_key = load("../experiments/results/ISO/bias$(bias)/correct/regions_key.jld")["regions_key"]
println()

# +
scatter_color = "blue"
estimate_color = "green"
truth_color = "red"

linewidth = 2
marker_size = 3
alpha = 0.2

# +
# Get Mean, Lower Bound, and Upper Bounds

estIntMean = Dict()
estIntLower = Dict()
estIntUpper = Dict()

nModels = length(keys(model_samples))
nStates = length(keys(model_samples["correct"]))

lower_bound = 0.005
upper_bound = 0.995

fig, axes = subplots(nStates, nModels, figsize=(10,10), constrained_layout=true, sharey="row")

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
        
#         index = (j-1) * nModels + i
        ax = axes[j, i]

        ax.fill_between(doTs * 100, 
                     estIntUpper[model][state] * 10000, 
                     estIntLower[model][state] * 10000, 
                     alpha = 0.1,
                     color = estimate_color)
        ax.plot(doTs * 100,
             estIntMean[model][state] * 10000,
             color = estimate_color,
             linewidth=linewidth,
             label = "Estimate")
        
        ax.plot(doTs*100,
             truthIntMeans[state]*10000, 
             color=truth_color, 
             linewidth=linewidth,
             label = "Ground Truth")
        
        in_state = regions_key .== state
        ax.scatter(allTs[state] * 100, 
            allYs[state] * 10000, 
            s=marker_size, 
            color=scatter_color, 
            alpha=alpha, 
            label="Heldout Data")
        ax.scatter(T[in_state] * 100, 
            Y[in_state] * 10000, 
            s=marker_size, 
            c=scatter_color, 
            marker="o", 
            label="Observed Data")

        ax.set_xlim(25, 75)
#         ylim(0, 100000)
        
        if i == 1
            ax.set_ylabel(state)
        end
        
        if j == 1
            ax.set_title(model_key[model])
        end
        
        ax.set_xticks([])
        ax.set_yticks([])
    end
end

handles, labels = axes[1,1].get_legend_handles_labels()
fig.legend(handles, labels, loc=8, bbox_to_anchor=[0.8, 0.13], ncol=2)

tight_layout(pad=0.4, w_pad=0.2, h_pad=0.2)
fig.text(0.5, -0.02, "Temperature", ha="center", va="center")
fig.text(-0.02, 0.5, "Energy Consumption", ha="center", va="center", rotation="vertical")
fig.savefig("../figures/NEED_multiples.png", dpi=200, bbox_inches="tight")
# -

