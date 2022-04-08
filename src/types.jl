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
    Vector{Int64},
    Vector{Bool},
    Vector{Float64},
    Array{Int64,1},
    Array{Bool,1},
}

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

SupportedRBFMatrix = Union{
    Matrix{Float64},
    Matrix{Int64},
    Matrix{Bool}
}

SupportedRBFLengthscale = Union{
    Float64,
    Array{Float64,1},
    FunctionalCollections.PersistentVector{Float64}
}

SupportedCovarianceMatrix = Union{
    Vector{Matrix{Float64}}
}