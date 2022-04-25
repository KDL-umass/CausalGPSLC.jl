@gen function prior(addr)
    if addr == :intercept
        return @trace(inv_gamma(2, 2), :intercept)
    elseif addr == :slope
        return @trace(normal(0, 1), :slope)
    end
end

@gen function proposal(trace, addr)
    if addr == :intercept
        return @trace(inv_gamma(2, 2), :intercept)
    elseif addr == :slope
        return @trace(normal(0, 1), :slope)
    end
end

"""Super simple test model with standard GPSLC model calling API"""
@gen function model(hyperparams, nU, X, T)
    theta = zeros(2) # linear model
    theta[1] = @trace(prior(:intercept))
    theta[2] = @trace(prior(:slope))
    n = size(X, 1)
    @trace(mvnormal(X .* theta[2] .+ theta[1], LinearAlgebra.I(n)), :Y)
end

function posterior(hyperparams::Nothing, X, T::Nothing, Y, nU::Nothing, nOuter,
    nMHInner::Nothing, nESInner::Nothing) # numSamples=nOuter
    obs = choicemap()
    obs[:Y] = Y
    (trace, _) = generate(model, (hyperparams, nU, X, T), obs)
    samples = []
    for i in 1:nOuter
        trace, _ = mh(trace, proposal, (:intercept,))
        trace, _ = mh(trace, proposal, (:slope,))
        push!(samples, get_choices(trace))
    end
    samples, trace
end

"""Get super simple linear model with GPSLC API structure"""
function getToyModel()
    return model, posterior
end