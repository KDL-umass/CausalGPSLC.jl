export generateSigmaU,
    removeAdjacent,
    toMatrix, toTupleOfVectors,
    getAddresses,
    getN, getNU, getNX,
    getNumPosteriorSamples,
    extractParameters

"""
    generateSigmaU(nIndividualsArray)
    generateSigmaU(nIndividualsArray, eps)
    generateSigmaU(nIndividualsArray, eps, cov)
Generate block matrix for U given object counts
    
SigmaU is shorthand for the object structure of the latent confounder
"""
function generateSigmaU(nIndividualsArray::Array{Int64},
    eps::Float64=1e-13, cov::Float64=1.0)::ConfounderStructure

    n = sum(nIndividualsArray)
    SigmaU = Matrix{Float64}(I, n, n)
    i = 1 # top left corner of block in block-matrix
    for nIndividuals in nIndividualsArray
        SigmaU[
            i:i+nIndividuals-1,
            i:i+nIndividuals-1
        ] = ones(nIndividuals, nIndividuals) * cov
        i += nIndividuals
    end

    SigmaU[diagind(SigmaU)] .= 1 + eps
    return SigmaU
end

"""
    removeAdjacent(vector)
Return vector where each element is distinct from the previous one
"""
function removeAdjacent(v)
    output = Vector{eltype(v)}()
    if length(v) > 0
        prevElement = v[1]
        push!(output, prevElement)
        for e in v
            if e != prevElement
                prevElement = e
                push!(output, prevElement)
            end
        end
    end
    return output
end


"""
    toMatrix(X, n, m)
Convert a vector of vectors or similar to a 2D matrix.
Only call if you know all subvectors are same length.
"""
function toMatrix(X::ReshapeableMatrix, n::Int64, m::Int64)
    X = permutedims(hcat(X...))
    X = reshape(X, (n, m))
    return X
end

"""
    toTupleOfVectors(matrix)
Convert matrix to tuple of vectors.
"""
function toTupleOfVectors(U::Union{Matrix{Bool},Matrix{Int64},Matrix{Float64}})
    Tuple(U[i, :] for i = 1:size(U, 1))
end

"""
    getAddresses(choicemap)
Debugging tool to print all available address keys in choicemap
"""
function getAddresses(choices::Gen.ChoiceMap)
    addresses = []
    for (addr, val) in get_values_shallow(choices)
        push!(addresses, addr)
    end
    return addresses
end

"""
    extractParameters(g, posteriorSampleIdx)

Get inferred parameters from `g`'s `posteriorSampleIdx`_th_ posterior sample. 
Parameters are `uyLS, xyLS, tyLS, yNoise, yScale, U`, some of which are allowed to be `Nothing`
"""
function extractParameters(g::GPSLCObject, posteriorSampleIdx::Int64)
    i = posteriorSampleIdx
    n = getN(g)
    nU = getNU(g)
    if nU === nothing
        uyLS = nothing
        U = nothing
    else
        uyLS = zeros(nU)
        U = zeros(n, nU)
        for u in 1:nU
            uyLS[u] = g.posteriorSamples[i][:uyLS=>u=>:LS]
            U[:, u] = g.posteriorSamples[i][:U=>u=>:U]
        end
        U = toMatrix(U, n, nU)
        @assert size(U) == (n, nU)
    end

    if g.X === nothing
        xyLS = nothing
    else
        nX = getNX(g)
        xyLS = zeros(nX)
        for k in 1:nX
            xyLS[k] = g.posteriorSamples[i][:xyLS=>k=>:LS]
        end
    end
    tyLS = g.posteriorSamples[i][:tyLS]
    yNoise = g.posteriorSamples[i][:yNoise]
    yScale = g.posteriorSamples[i][:yScale]

    return uyLS, xyLS, tyLS, yNoise, yScale, U
end

"""
    getN(g)
Number of individuals.
"""
function getN(g::GPSLCObject)
    size(g.Y, 1)
end

"""
    getNX(g)
Number of covariates (and observed confounders).
"""
function getNX(g::GPSLCObject)
    size(g.X, 2)
end

"""
    getNU(g)    
Number of latent confounders to perform inference over.
"""
function getNU(g::GPSLCObject)
    g.hyperparams.nU
end

"""
    getNumPosteriorSamples(g)
Number of posterior samples that will be used based on hyperparameters.
"""
function getNumPosteriorSamples(g::GPSLCObject)
    burnIn = g.hyperparams.nBurnIn
    stepSize = g.hyperparams.stepSize
    nOuter = g.hyperparams.nOuter
    length(burnIn:stepSize:nOuter)
end