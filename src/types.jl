export HyperParameters,
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

"""Global hyperparameters"""
HyperParameters = Dict{String,Any}

"""Confounder (U)"""
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


"""Covariates (X)"""
Covariates = Union{
    Array{Array{Float64,1}},
    Array{Vector{Float64}},
    Vector{Vector{Float64}},
    Array{Float64,2},
    Matrix{Float64},
}

"""Treatment (T)"""
Treatment = Union{
    Vector{Bool},
    Vector{Float64},
    FunctionalCollections.PersistentVector{Bool},
    FunctionalCollections.PersistentVector{Float64},
}

"""Outcome (Y)"""
Outcome = Union{
    Vector{Float64},
}


"""Viable inputs to the rbfKernelLog function in linear algebra datatypes"""
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

"""Viable inputs to the rbfKernelLog function that are nested lists"""
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

"""Viable inputs to the rbfKernelLog function in linear algebra datatypes"""
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

"""Viable inputs to the rbfKernelLog function as kernel lengthscales"""
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

"""Viable inputs to the processCov function"""
SupportedCovarianceMatrix = Union{
    Vector{Matrix{Float64}},
}

XScaleOrNoise = Union{
    Vector{Vector{Float64}},
    FunctionalCollections.PersistentVector{Float64},
}

"""Matrix that can be reshaped"""
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