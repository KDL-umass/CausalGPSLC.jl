using GPSLC

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

"""
Super simple continuous treatment model with standard GPSLC model calling API
No confounders, no covariates
"""
@gen function model(priorparams::GPSLC.PriorParameters, n::Int64, nU::Nothing, nX::Nothing)
    theta = zeros(2) # linear model
    theta[1] = @trace(prior(:intercept))
    theta[2] = @trace(prior(:slope))
    T = @trace(generateRealTfromPrior(priorparams, n))
    cov = Matrix{Float64}(I, n, n)
    @trace(mvnormal(T .* theta[2] .+ theta[1], cov), :Y)
end

function posterior(priorparams::GPSLC.PriorParameters, X::Nothing, T::GPSLC.ContinuousTreatment, Y::GPSLC.Outcome, nU::Nothing, nOuter,
    nMHInner::Nothing, nESInner::Nothing) # numSamples=nOuter
    obs = choicemap()
    obs[:Y] = Y

    n = size(T, 1)
    nX = nothing

    (trace, _) = generate(model, (priorparams, n, nU, nX), obs)
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