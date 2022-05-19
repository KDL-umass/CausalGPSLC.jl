export predictCounterfactualEffects

"""
    predictCounterfactualOutcomes(g, nSamplesPerMixture)
    predictCounterfactualOutcomes(g, nSamplesPerMixture; fidelity=100)
    predictCounterfactualOutcomes(g, nSamplesPerMixture; fidelity=100, minDoT=0, maxDoT=5)

Params
- `g::`[`GPSLCObject`](@ref): The `GPSLCObject` that inference has already been computed for.
- `nSamplesPerMixture::Int64`: The number of outcome samples to 
draw from each set of inferred posterior parameters.
- `fidelity::Int64`: How many intervention values to use to cover the domain of treatment values. Higher means more samples.
- `minDoT::Float64=min(g.T...)`: The lowest interventional treatment to use.Defaults to the data `g.T`'s lowest treatment value.
- `maxDoT::Float64=max(g.T...)`: The highest interventional treatment to use. Defaults to the data `g.T`'s highest treatment value.

```julia
julia> ite, doT = predictCounterfactualEffects(g, 30; fidelity=100)
```

Returns 
- `ite::Matrix{Float64}`: An array of size `[d, n, numPosteriorSamples * nSamplesPerMixture]` where d is the number of interventional values defined by `fidelity` and the range of treatments in `g.T` - `doTrange::Vector{Float64}`: The list values of doT used, in order that matches the rows of `ite`.
"""
function predictCounterfactualEffects(g::GPSLCObject, nSamplesPerMixture::Int64; fidelity::Int64=100, minDoT=min(g.T...), maxDoT=max(g.T...))
    delta = abs(maxDoT - minDoT)
    nps = getNumPosteriorSamples(g)
    numSamples = nps * nSamplesPerMixture
    step = delta / fidelity
    doTrange = minDoT:step:maxDoT

    ite = zeros(length(doTrange), getN(g), numSamples)
    for (i, doT) in enumerate(doTrange)
        ite[i, :, :] = sampleITE(g, doT; samplesPerPosterior=nSamplesPerMixture)
    end

    return ite, doTrange
end
