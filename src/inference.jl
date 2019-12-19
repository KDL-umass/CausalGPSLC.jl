module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export ContinuousPosterior, BinaryPosterior

# +
#   Like a gaussian drift, we match the moments of our proposal
#   with the previous noise sample with a fixed variance.
#   See https://arxiv.org/pdf/1605.01019.pdf.

@gen (static) function uNoiseProposal(trace, var::Float64)
    cur = trace[:uNoise]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uNoise)
end

@gen (static) function tNoiseProposal(trace, var::Float64)
    cur = trace[:tNoise]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tNoise)
end

@gen (static) function yNoiseProposal(trace, var::Float64)
    cur = trace[:yNoise]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :yNoise)
end

@gen (static) function utLSProposal(trace, var::Float64)
    cur = trace[:utLS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :utLS)
end

@gen (static) function uyLSProposal(trace, var::Float64)
    cur = trace[:uyLS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uyLS)
end

@gen (static) function tyLSProposal(trace, var::Float64)
    cur = trace[:tyLS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tyLS)
end

@gen (static) function xtLSProposal(trace, i::Int, var::Float64)
    cur = trace[:xtLS => i => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :xtLS => i => :LS)
end

@gen (static) function xyLSProposal(trace, i::Int, var::Float64)
    cur = trace[:xyLS => i => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :xyLS => i => :LS)
end

@gen (static) function tScaleProposal(trace, var::Float64)
    cur = trace[:tScale]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tScale)
end

@gen (static) function yScaleProposal(trace, var::Float64)
    cur = trace[:yScale]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :yScale)
end

# +
load_generated_functions()

function ContinuousPosterior(hyperparams::Dict, T::Array{Float64}, Y::Array{Float64}, 
                                Xcol, nX::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    n = length(T)
    
    PosteriorSamples = []
    
    (trace, _) = generate(ContinuousGPROC, (hyperparams, Xcol, nX), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, utLSProposal, (0.5, ))
            (trace, _) = mh(trace, uyLSProposal, (0.5, ))
            (trace, _) = mh(trace, tyLSProposal, (0.5, ))
            
            for k=1:nX
                (trace, _) = mh(trace, xtLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xyLSProposal, (k, 0.5))
            end
            
            (trace, _) = mh(trace, tScaleProposal, (0.5, ))
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))    
        end
             
        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]
        
        for j=1:nESInner
            trace = elliptical_slice(trace, :U, zeros(n), uCov)
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

function BinaryPosterior(hyperparams::Dict, T::Array{Float64}, Y::Array{Float64}, 
                                Xcol, nX::Int, nOuter::Int, nMHInner::Int, nESInner::Int)
    load_generated_functions()
    
    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    n = length(T)
    
    PosteriorSamples = []
    
    (trace, _) = generate(BinaryGPROC, (hyperparams, Xcol, nX), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, utLSProposal, (0.5, ))
            (trace, _) = mh(trace, uyLSProposal, (0.5, ))
            (trace, _) = mh(trace, tyLSProposal, (0.5, ))
            
            for k=1:nX
                (trace, _) = mh(trace, xtLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xyLSProposal, (k, 0.5))
            end
            
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))
        end
        
        choices = get_choices(trace)
        
        U = choices[:U]
        utLS = choices[:utLS]
        xtLS = choices[:xtLS]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]
        
        logitTCov = (broadcast(hyperparams["utKernel"], U, U', utLS) .* 
                     broadcast(hyperparams["xtKernel"], Xcol, (xtLS,)) * tScale) + tNoise * 1I        
        uCov = hyperparams["SigmaU"] * uNoise
        
        for j=1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTCov)
            trace = elliptical_slice(trace, :U, zeros(n), uCov)
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end
# -


end
