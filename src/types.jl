export Covariates, HyperParameters, SupportedRBFVector

"""Covariates (X)"""
Covariates = Union{
    Array{Array{Float64,1}},
    Array{Vector{Float64}},
    Vector{Vector{Float64}},
    Array{Float64,2},
    Matrix{Float64},
}

"""Global hyperparameters"""
HyperParameters = Dict{String,Any}

"""Viable inputs to the rbfKernelLog function"""
SupportedRBFVector = Union{
    FunctionalCollections.PersistentVector{Float64},
    FunctionalCollections.PersistentVector{Bool},
    Array{Float64,1},
    Vector{Float64},
    Array{Int64,1},
    Vector{Int64},
    Array{Bool,1},
    Vector{Bool},
}

SupportedRBFData = Union{
    FunctionalCollections.PersistentVector{SupportedRBFVector},
    Vector{SupportedRBFVector},
    Array{SupportedRBFVector,1}
}

SupportedRBFMatrix = Union{
    Matrix{Float64},
    Matrix{Int64},
    Matrix{Bool}
}

SupportedRBFLengthscale = Union{
    Array{Float64,1},
    FunctionalCollections.PersistentVector{Float64}
}