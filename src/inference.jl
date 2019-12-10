module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export AdditiveNoisePosterior
# -

u_selection = StaticSelection(select(:U))


function AdditiveNoisePosterior(hyperparams, X, Y, nOuter, nMHInner, nESInner)
    obs = Gen.choicemap()
    obs[:X] = X
    obs[:Y] = Y
    
    noise_selection = StaticSelection(select(:uNoise, :xNoise, :yNoise))
    LS_selection = StaticSelection(select(:uxLS, :uyLS, :xyLS))
    scale_selection = StaticSelection(select(:xScale, :yScale))
    
    n = length(X)
    
    PosteriorSamples = []
    
    (tr, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nOuter    
        for j=1:nMHInner
            (tr, _) = mh(tr, noise_selection)
            (tr, _) = mh(tr, LS_selection)
            (tr, _) = mh(tr, scale_selection)
        end
        
        uCov = hyperparams["SigmaU"] * get_choices(tr)[:uNoise]
        
        for k=1:nESInner
            tr = elliptical_slice(tr, :U, zeros(n), uCov)
        end
        
        push!(PosteriorSamples, get_choices(tr))
    end
    PosteriorSamples, tr
end


# +
# TODO: Test this function 

# TODO: Change Us to include epsY. We don't need epsX for 
# estimating any causal estimates, but we do need to take mh steps
# over epsX to make sure we sample from the marginal of epsY.

function Posterior(hyperparams, X, Y, nSteps, nESS)
    obs = Gen.choicemap()
    obs[:X] = X
    obs[:Y] = Y
    
    n = length(X)
    
    Us = zeros(nSteps, n)
    
    (tr, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    
    for i=1:nSteps
        # Update U w/ ESS
        for j=1:nESS
            tr = elliptical_slice(tr, :U, zeros(n), hyperparams["uCov"])
        end

        # Update epsX and epsY with generic mh
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
