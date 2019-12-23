module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export Posterior

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

@gen (static) function xNoiseProposal(trace, i::Int, var::Float64)
    cur = trace[:xNoise => i => :Noise]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :xNoise => i => :Noise)
end

@gen (static) function yNoiseProposal(trace, var::Float64)
    cur = trace[:yNoise]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :yNoise)
end

@gen (static) function utLSProposal(trace, i::Int, var::Float64)
    cur = trace[:utLS => i => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :utLS => i => :LS)
end

@gen (static) function uyLSProposal(trace, i::Int, var::Float64)
    cur = trace[:uyLS => i => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uyLS => i => :LS)
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

@gen (static) function uxLSProposal(trace, i::Int, j::Int, var::Float64)
    cur = trace[:uxLS => i => j => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uxLS => i => j => :LS)
end

@gen (static) function xyLSProposal(trace, i::Int, var::Float64)
    cur = trace[:xyLS => i => :LS]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :xyLS => i => :LS)
end

@gen (static) function xScaleProposal(trace, i::Int, var::Float64)
    cur = trace[:xScale => i => :Scale]
    
    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :xScale => i => :Scale)
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

function Posterior(hyperparams::Dict, X::Array{Array{Float64, 1}}, T::Array{Float64}, Y::Array{Float64}, 
                   nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)
    nX = length(X)
    
    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y
    
    for i in 1:nX
        obs[:X => i => :X] = X[i]
    end
    
    PosteriorSamples = []
    
    (trace, _) = generate(ContinuousGPROC, (hyperparams, nX, nU), obs)
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tyLSProposal, (0.5, ))
            
            for k=1:nU
                (trace, _) = mh(trace, utLSProposal, (k, 0.5))
                (trace, _) = mh(trace, uyLSProposal, (k, 0.5))
                for l=1:nX
                    (trace, _) = mh(trace, uxLSProposal, (k, l, 0.5))
                end
            end
            
            for k=1:nX
                (trace, _) = mh(trace, xNoiseProposal, (k, 0.5))
                (trace, _) = mh(trace, xtLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xyLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xScaleProposal, (k, 0.5))
            end
            
            (trace, _) = mh(trace, tScaleProposal, (0.5, ))
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))    
        end
             
        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]
        
        for j=1:nESInner
            for k=1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

function Posterior(hyperparams::Dict, X::Array{Array{Float64, 1}}, T::Array{Bool}, Y::Array{Float64}, 
                   nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)
    
    n = length(T)
    nX = length(X)
    
    obs = Gen.choicemap()
    
    obs[:Y] = Y
    for i in 1:n
        obs[:T => i => :T] = T[i]
    end
    
    for i in 1:nX
        obs[:X => i => :X] = X[i]
    end
    
    PosteriorSamples = []
    
    (trace, _) = generate(BinaryGPROC, (hyperparams, nX, nU), obs)
    for i=1:nOuter    
        println(i)
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tyLSProposal, (0.5, ))
            
            for k=1:nU
                (trace, _) = mh(trace, utLSProposal, (k, 0.5))
                (trace, _) = mh(trace, uyLSProposal, (k, 0.5))
                for l=1:nX
                    (trace, _) = mh(trace, uxLSProposal, (k, l, 0.5))
                end
            end
            
            for k=1:nX
                (trace, _) = mh(trace, xNoiseProposal, (k, 0.5))
                (trace, _) = mh(trace, xtLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xyLSProposal, (k, 0.5))
                (trace, _) = mh(trace, xScaleProposal, (k, 0.5))
            end
            
            (trace, _) = mh(trace, tScaleProposal, (0.5, ))
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))    
        end
        
        choices = get_choices(trace)
        
        U = [choices[:U => i => :U] for i in 1:nU]
        utLS = [choices[:utLS => i => :LS] for i in 1:nU]
        xtLS = [choices[:xtLS => i => :LS] for i in 1:nX]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]
        
        utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
        xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
        logitTcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
       
        uCov = hyperparams["SigmaU"] * uNoise
        
        for j=1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTcov)
            for k=1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end
# -


end
