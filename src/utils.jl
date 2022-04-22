export generateSigmaU, removeAdjacent, toMatrix, toTupleOfVectors

"""
Generate block matrix for U given object counts
    
SigmaU is shorthand for the object structure of the latent confounder
"""
function generateSigmaU(nIndividualsArray::Array{Int64},
    eps::Float64=1e-13, cov::Float64=1.0)

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
function toMatrix(U::ReshapeableMatrix, n::Int64, m::Int64)
    U = permutedims(hcat(U...))
    U = reshape(U, (n, m))
    return U
end

"""Convert matrix to list of lists"""
function toTupleOfVectors(U::Union{Matrix{Bool},Matrix{Int64},Matrix{Float64}})
    Tuple(U[i, :] for i = 1:size(U, 1))
end