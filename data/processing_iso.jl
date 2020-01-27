module ProcessingISO

# +
using DataFrames
using Statistics
using StatsBase
using KernelDensity
using Distributions

include("synthetic.jl")
using .Synthetic
# -

export generateImportanceWeights, resampleData

# +
# for region in regions
#     is_region = weekend_df[!, :Region] .== region
#     hist(weekend_df[is_region, :][!, :DryBulbTemp], density=true, label=region)
# end

# legend()
# There is some confounding, as more northern states have slightly colder temperatures. 
# This effect is pretty minimal though.

# +
# for region in regions
#     is_region = weekday_df[!, :Region] .== region
#     scatter(weekday_df[is_region, :][!, :DryBulbTemp], 
#             weekday_df[is_region, :][!, :RealTimeDemand], label=region)
# end

# legend()
# -

function generateImportanceWeights(newMeans, newVars, weekday_df)
    importanceWeights = Dict()

    for state in keys(newMeans)
        is_state = weekday_df[!, :State] .== state
        temp_data = weekday_df[is_state, :][!, :DryBulbTemp]
        ik = InterpKDE(kde(temp_data))
        oldPDFs = pdf(ik, temp_data)
        newDist = Truncated(Normal(newMeans[state], newVars[state]), minimum(temp_data), maximum(temp_data))
        newPDFs = pdf.(newDist, temp_data)
        importanceWeights[state] = Weights(newPDFs ./ oldPDFs)
    end
    
    return importanceWeights
end

function resampleData(nSamplesPerState, importanceWeights, weekday_df)
    indeces = [i for i in 1:length(importanceWeights["CT"])]
    states = keys(importanceWeights)
    n = length(states) * nSamplesPerState
    
    T = zeros(n)
    Y = zeros(n)
    
    i = 1
    
    for state in states
        is_state = weekday_df[!, :State] .== state
        state_data = weekday_df[is_state, :]
        is_sampled = sample(indeces, importanceWeights[state], nSamplesPerState, replace=false)
        sampled_data = state_data[is_sampled, :]
        T[i:i+nSamplesPerState-1] = sampled_data[!, :DryBulbTemp]
        Y[i:i+nSamplesPerState-1] = sampled_data[!, :RealTimeDemand]
        i += nSamplesPerState
    end

    eps = 1e-7
    cov = 1.
    nIndividualsArray = [nSamplesPerState for i in 1:length(states)]
    sigmaU = generateSigmaU(n, nIndividualsArray, eps, cov)  
    states_key = vcat(fill.(states, nSamplesPerState)...)
    
    return T, Y, sigmaU, states_key
end

end
