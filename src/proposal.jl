export getProposalAddress, paramProposal

"""
Optimizes `paramProposal` by providing compact way to access trace address symbols
"""
function getProposalAddress(name::String; i::Int = -1, j::Int = -1)
    proposalAddresses = Dict(
        "uNoise" => :uNoise,
        "tNoise" => :tNoise,
        "xNoise" => (:xNoise => i => :Noise),
        "yNoise" => :yNoise,
        "utLS" => (:utLS => i => :LS),
        "uyLS" => (:uyLS => i => :LS),
        "tyLS" => :tyLS,
        "xtLS" => (:xtLS => i => :LS),
        "uxLS" => (:uxLS => i => j => :LS),
        "xyLS" => (:xyLS => i => :LS),
        "xScale" => (:xScale => i => :Scale),
        "tScale" => :tScale,
        "yScale" => :yScale,
    )
    return proposalAddresses[name]
end

"""
Like a Gaussian drift, we match the moments of our proposal
with the previous noise sample with a fixed variance.
See https://arxiv.org/pdf/1605.01019.pdf.
"""
@gen function paramProposal(trace, variance::Float64, addr)
    cur = trace[addr]

    # inv_gamma parameters, centered at cur with known variance
    shape = (cur * cur / variance) + 2
    scale = cur * (shape - 1)

    # Prior is `inv_gamma` [0,+inf]
    @trace(inv_gamma(shape, scale), addr)
end
