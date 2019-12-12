module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export AdditiveNoisePosterior, LinearAdditiveNoisePosterior
# -

u_selection = StaticSelection(select(:U))


# +
#   Like a gaussian drift, we match the moments of our proposal
#   with the previous noise sample with a fixed variance.
#   See https://arxiv.org/pdf/1605.01019.pdf.

@gen (static) function uNoiseProposal(trace, var)
    Noise = trace[:uNoise]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uNoise)
end

@gen (static) function tNoiseProposal(trace, var)
    Noise = trace[:tNoise]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tNoise)
end

@gen (static) function yNoiseProposal(trace, var)
    Noise = trace[:yNoise]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :yNoise)
end

@gen (static) function utLSProposal(trace, var)
    Noise = trace[:utLS]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :utLS)
end

@gen (static) function uyLSProposal(trace, var)
    Noise = trace[:uyLS]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :uyLS)
end

@gen (static) function tyLSProposal(trace, var)
    Noise = trace[:tyLS]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tyLS)
end

@gen (static) function tScaleProposal(trace, var)
    Noise = trace[:tScale]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :tScale)
end

@gen (static) function yScaleProposal(trace, var)
    Noise = trace[:yScale]
    
    Shape = (Noise * Noise / var) + 2
    Scale = Noise * (Shape - 1)
    
    @trace(inv_gamma(Shape, Scale), :yScale)
end
# -

function AdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
    load_generated_functions()
    
    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    n = length(T)
    
    PosteriorSamples = []
    
    (trace, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, utLSProposal, (0.5, ))
            (trace, _) = mh(trace, uyLSProposal, (0.5, ))
            (trace, _) = mh(trace, tyLSProposal, (0.5, ))
            (trace, _) = mh(trace, tScaleProposal, (0.5, ))
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))
            
        end
        
        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]
        
        for k=1:nESInner
            trace = elliptical_slice(trace, :U, zeros(n), uCov)
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end


function LinearAdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
    load_generated_functions()
    
    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    n = length(T)
    
    PosteriorSamples = []
    
    (trace, _) = generate(LinearAdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, uNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, tNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, yNoiseProposal, (0.5, ))
            (trace, _) = mh(trace, utLSProposal, (0.5, ))
            (trace, _) = mh(trace, uyLSProposal, (0.5, ))
            (trace, _) = mh(trace, tScaleProposal, (0.5, ))
            (trace, _) = mh(trace, yScaleProposal, (0.5, ))
        end
        
        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]
        
        for k=1:nESInner
            trace = elliptical_slice(trace, :U, zeros(n), uCov)
        end
        
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

# +
# TODO: Test this function 

# TODO: Change Us to include epsY. We don't need epsT for 
# estimating any causal estimates, but we do need to take mh steps
# over epsX to make sure we sample from the marginal of epsY.

function Posterior(hyperparams, T, Y, nSteps, nESS)
    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    n = length(T)
    
    Us = zeros(nSteps, n)
    
    (tr, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nSteps
        # Update U w/ ESS
        for j=1:nESS
            tr = elliptical_slice(tr, :U, zeros(n), hyperparams["uCov"])
        end

        # Update epsT and epsY with generic mh
        for k=1:n
            (tr, _) = mh(tr, select(:epsX => k => :eps))
            (tr, _) = mh(tr, select(:epsY => k => :eps))
        end

        Us[iter, :] = get_choices(tr)[:U]
    end
    Us, tr
end
# -

end
