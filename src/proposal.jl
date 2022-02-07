module Proposal

using Gen
using ProgressBars


#   Like a gaussian drift, we match the moments of our proposal
#   with the previous noise sample with a fixed variance.
#   See https://arxiv.org/pdf/1605.01019.pdf.

export uNoiseProposal
@gen function uNoiseProposal(trace, var::Float64)
    cur = trace[:uNoise]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :uNoise)
end

export tNoiseProposal
@gen function tNoiseProposal(trace, var::Float64)
    cur = trace[:tNoise]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :tNoise)
end

export xNoiseProposal
@gen function xNoiseProposal(trace, i::Int, var::Float64)
    cur = trace[:xNoise=>i=>:Noise]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :xNoise => i => :Noise)
end

export yNoiseProposal
@gen function yNoiseProposal(trace, var::Float64)
    cur = trace[:yNoise]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :yNoise)
end

export utLSProposal
@gen function utLSProposal(trace, i::Int, var::Float64)
    cur = trace[:utLS=>i=>:LS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :utLS => i => :LS)
end

export uyLSProposal
@gen function uyLSProposal(trace, i::Int, var::Float64)
    cur = trace[:uyLS=>i=>:LS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :uyLS => i => :LS)
end

export tyLSProposal
@gen function tyLSProposal(trace, var::Float64)
    cur = trace[:tyLS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :tyLS)
end

export xtLSProposal
@gen function xtLSProposal(trace, i::Int, var::Float64)
    cur = trace[:xtLS=>i=>:LS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :xtLS => i => :LS)
end

export uxLSProposal
@gen function uxLSProposal(trace, i::Int, j::Int, var::Float64)
    cur = trace[:uxLS=>i=>j=>:LS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :uxLS => i => j => :LS)
end

export xyLSProposal
@gen function xyLSProposal(trace, i::Int, var::Float64)
    cur = trace[:xyLS=>i=>:LS]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :xyLS => i => :LS)
end

export xScaleProposal
@gen function xScaleProposal(trace, i::Int, var::Float64)
    cur = trace[:xScale=>i=>:Scale]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :xScale => i => :Scale)
end

export tScaleProposal
@gen function tScaleProposal(trace, var::Float64)
    cur = trace[:tScale]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :tScale)
end

export yScaleProposal
@gen function yScaleProposal(trace, var::Float64)
    cur = trace[:yScale]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :yScale)
end

load_generated_functions()

end