export generateSigmaU, removeAdjacent, toMatrix

"""
Generate block matrix for U given object counts
    
SigmaU is shorthand for the object structure of the latent confounder
"""
function generateSigmaU(nIndividualsArray::Array{Int},
    eps::Float64=1e-13, cov::Float64=1.0)

    n = sum(nIndividualsArray)
    SigmaU = Matrix{Float64}(I, n, n)
    i = 1
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

"""Return vector where each element is distinct from the previous one"""
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


"""Only call if you know all subvectors are same length"""
function toMatrix(U::Union{
        Vector{Any},
        Vector{Vector{Bool}},
        Vector{Vector{Int64}},
        Vector{Vector{Float64}},
        FunctionalCollections.PersistentVector{Vector{Bool}},
        FunctionalCollections.PersistentVector{Vector{Int64}},
        FunctionalCollections.PersistentVector{Vector{Float64}}
    }, n::Int64, m::Int64)
    U = permutedims(hcat(U...))
    U = reshape(U, (n, m))
    return U
end
