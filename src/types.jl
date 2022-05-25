export GPSLCObject,
    HyperParameters,
    areequal,
    PriorParameters,
    ConfounderStructure,
    Confounders,
    Covariates,
    Treatment,
    Outcome,
    SupportedRBFVector,
    SupportedRBFData,
    SupportedRBFMatrix,
    SupportedRBFLengthscale,
    SupportedCovarianceMatrix,
    XScaleOrNoise,
    ReshapeableMatrix

"""
    HyperParameters
Define the high-level attributes of the inference procedure. More information on each of the attributes can be found in [`getHyperParameters`](@ref).
"""
mutable struct HyperParameters
    nU::Union{Int64,Nothing}
    nOuter::Int64
    nMHInner::Union{Int64,Nothing}
    nESInner::Union{Int64,Nothing}
    nBurnIn::Int64
    stepSize::Int64
    predictionCovarianceNoise::Float64
end

import Base.==
function (==)(a::HyperParameters, b::HyperParameters)
    nU = a.nU == b.nU
    nOuter = a.nOuter == b.nOuter
    nMHInner = a.nMHInner == b.nMHInner
    nESInner = a.nESInner == b.nESInner
    nBurnIn = a.nBurnIn == b.nBurnIn
    stepSize = a.stepSize == b.stepSize
    predictionCovarianceNoise = a.predictionCovarianceNoise == b.predictionCovarianceNoise

    @assert nU "nU didn't match"
    @assert nOuter "nOuter didn't match"
    @assert nMHInner "nMHInner didn't match"
    @assert nESInner "nESInner didn't match"
    @assert nBurnIn "nBurnIn didn't match"
    @assert stepSize "stepSize didn't match"
    @assert predictionCovarianceNoise "predictionCovarianceNoise didn't match"

    (nU &&
     nOuter &&
     nMHInner &&
     nESInner &&
     nBurnIn &&
     stepSize &&
     predictionCovarianceNoise)
end


"""
    PriorParameters
Contains shapes and scales for various Inverse Gamma distributions used as priors for kernel parameters and other parameters. More information on each of the attributes can be found in [`getPriorParameters`](@ref).
"""
PriorParameters = Dict{String,Any}

"""
    SigmaU 
structured prior for U.
"""
ConfounderStructure = Matrix{Float64}

"""
    Object Labels for instances (obj)
    
Optional for GPSLC, but per publication it improves performance.
"""
ObjectLabels = Any

"""
    Confounders (U)
Latent confounders that GPSLC performs inference over.

Either 1D or 2D matrices of `Float64` values.
"""
Confounders = Union{
    Array{Array{Float64,1}},
    Array{Vector{Float64}},
    Vector{Vector{Float64}},
    Array{Float64,2},
    Matrix{Float64},
    Vector{Float64},
    FunctionalCollections.PersistentVector{Vector{Float64}},
    FunctionalCollections.PersistentVector{Float64},
}


"""
   Covariates (X)
Observed confounders and covariates.

`Matrix{Float64}` is the only valid structure for covariates
"""
Covariates = Union{
    Matrix{Float64},
}

"""Binary Treatment (T)"""
BinaryTreatment = Union{
    Vector{Bool},
    FunctionalCollections.PersistentVector{Bool},
}

"""Continuous Treatment (T)"""
ContinuousTreatment = Union{
    Vector{Float64},
    FunctionalCollections.PersistentVector{Float64},
}

"""
    Treatment (T)
Is made up of `BinaryTreatment` which is an alias for `Vector{Bool}` and `ContinuousTreatment` which is an alias for `Vector{Float64}`.
These types support other vector types to afford compatibility with internal libraries.
"""
Treatment = Union{
    BinaryTreatment,
    ContinuousTreatment
}

"""
    Outcome (Y)
The outcome for the series of Gaussian Process predictions is a `Vector{Float64}`. Currently only continuous values are supported as outcomes for input data.
"""
Outcome = Union{
    Vector{Float64},
}

"""Intervention (doT)"""
Intervention = Union{
    Bool,
    Vector{Bool},
    Float64,
    Vector{Float64},
}


"""
    SupportedRBFVector
Viable inputs to the rbfKernelLog function in linear algebra datatypes.
"""
SupportedRBFVector = Union{
    FunctionalCollections.PersistentVector{Float64},
    FunctionalCollections.PersistentVector{Bool},
    Array{Float64,1},
    Vector{Int64},
    Vector{Bool},
    Vector{Float64},
    Array{Int64,1},
    Array{Bool,1},
}

"""
    SupportedRBFData
Viable inputs to the rbfKernelLog function that are nested lists.
"""
SupportedRBFData = Union{
    FunctionalCollections.PersistentVector{Vector{Float64}},
    FunctionalCollections.PersistentVector{Vector{Int64}},
    FunctionalCollections.PersistentVector{Vector{Bool}},
    Vector{Vector{Float64}},
    Vector{Vector{Int64}},
    Vector{Vector{Bool}},
    Array{Vector{Float64},1},
    Array{Vector{Int64},1},
    Array{Vector{Bool},1}
}

"""
    SupportedRBFMatrix
Viable inputs to the rbfKernelLog function in linear algebra datatypes.
"""
SupportedRBFMatrix = Union{
    Matrix{Float64},
    Matrix{Int64},
    Matrix{Bool},
    Vector{Float64},
    Vector{Int64},
    Vector{Bool},
    FunctionalCollections.PersistentVector{Float64},
    FunctionalCollections.PersistentVector{Int64},
    FunctionalCollections.PersistentVector{Bool},
}

"""
    SupportedRBFLengthscale
Viable inputs to the rbfKernelLog function as kernel lengthscales.
"""
SupportedRBFLengthscale = Union{
    Int64,
    Matrix{Int64},
    Vector{Int64},
    Array{Int64,1},
    FunctionalCollections.PersistentVector{Int64},
    Float64,
    Matrix{Float64},
    Vector{Float64},
    Array{Float64,1},
    FunctionalCollections.PersistentVector{Float64},
}

"""
    SupportedCovarianceMatrix
Viable inputs to the processCov function.
"""
SupportedCovarianceMatrix = Union{
    Vector{Matrix{Float64}},
}

XScaleOrNoise = Union{
    Vector{Vector{Float64}},
    FunctionalCollections.PersistentVector{Float64},
}

"""
    ReshapeableMatrix
Matrix that can be reshaped.
"""
ReshapeableMatrix = Union{
    Matrix{Bool},
    Matrix{Int64},
    Matrix{Float64},
    Vector{Vector{Bool}},
    Vector{Vector{Int64}},
    Vector{Vector{Float64}},
    FunctionalCollections.PersistentVector{Vector{Bool}},
    FunctionalCollections.PersistentVector{Vector{Int64}},
    FunctionalCollections.PersistentVector{Vector{Float64}},
    FunctionalCollections.PersistentVector{FunctionalCollections.PersistentVector{Bool}},
    FunctionalCollections.PersistentVector{FunctionalCollections.PersistentVector{Int64}},
    FunctionalCollections.PersistentVector{FunctionalCollections.PersistentVector{Float64}},
}

"""
    GPSLCObject

This is the struct in GPSLC.jl that contains the data, hyperparamters, prior parameters, and posterior samples. It provides the primary interfaces to abstract the internals of GPSLC away from the higher-order functions like [`sampleITE`](@ref), [`sampleSATE`](@ref), and [`predictCounterfactualEffects`](@ref).

Returned by [`gpslc`](@ref)
"""
struct GPSLCObject
    hyperparams::HyperParameters
    priorparams::PriorParameters
    SigmaU::Union{ConfounderStructure,Nothing}
    obj::Union{ObjectLabels,Nothing}
    X::Union{Covariates,Nothing}
    T::Treatment
    Y::Outcome
    posteriorSamples::Vector{Any}
end

"""
Constructor for GPSLCObject that samples from the 
posterior before constructing the GPSLCObject.

    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, nothing, T, Y)
    GPSLCObject(hyperparams, priorparams, nothing, nothing, X, T, Y)
    GPSLCObject(hyperparams, priorparams, nothing, nothing, nothing, T, Y)

Full Model or model with no observed Covariates
"""
function GPSLCObject(hyperparams::HyperParameters, priorparams::PriorParameters, SigmaU::ConfounderStructure, obj::ObjectLabels, X::Union{Covariates,Nothing}, T::Treatment, Y::Outcome)
    posteriorSamples = samplePosterior(hyperparams, priorparams, SigmaU, X, T, Y)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y, posteriorSamples)
end

"""No Confounders"""
function GPSLCObject(hyperparams::HyperParameters, priorparams::PriorParameters, SigmaU::Nothing, obj::Nothing, X::Covariates, T::Treatment, Y::Outcome)
    hyperparams.nU = nothing
    posteriorSamples = samplePosterior(hyperparams, priorparams, SigmaU, X, T, Y)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y, posteriorSamples)
end

"""No Confounders, No Covariates"""
function GPSLCObject(hyperparams::HyperParameters, priorparams::PriorParameters, SigmaU::Nothing, obj::Nothing, X::Nothing, T::Treatment, Y::Outcome)
    hyperparams.nU = nothing
    hyperparams.nMHInner = nothing
    hyperparams.nESInner = nothing
    posteriorSamples = samplePosterior(hyperparams, priorparams, SigmaU, X, T, Y)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y, posteriorSamples)
end
