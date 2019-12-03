module Inference

# +
using Gen

include("model.jl")
using .Model

import Base.show
export AdditiveNoisePosterior, Posterior
# -

u_selection = StaticSelection(select(:U))


function AdditiveNoisePosterior(hyperparams, X, Y, nSteps)
    obs = Gen.choicemap()
    obs[:X] = X
    obs[:Y] = Y
    
    n = length(X)
    
    Us = zeros(nSteps, n)
    
    (tr, _) = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
    for iter=1:nSteps
        tr = elliptical_slice(tr, :U, zeros(n), hyperparams["uCov"])
        Us[iter, :] = get_choices(tr)[:U]
    end
    Us, tr
end


# +
# TODO, Add eps samples

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
