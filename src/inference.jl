module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export AdditiveNoisePosterior
# -

u_selection = StaticSelection(select(:U))


function AdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
    obs = Gen.choicemap()
    obs[:Tr] = T
    obs[:Y] = Y
    
    noise_selection = StaticSelection(select(:uNoise, :tNoise, :yNoise))
    LS_selection = StaticSelection(select(:utLS, :uyLS, :tyLS))
    scale_selection = StaticSelection(select(:tScale, :yScale))
    
    n = length(T)
    
    PosteriorSamples = []
    
    (trace, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (trace, _) = mh(trace, noise_selection)
            (trace, _) = mh(trace, LS_selection)
            (trace, _) = mh(trace, scale_selection)
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
