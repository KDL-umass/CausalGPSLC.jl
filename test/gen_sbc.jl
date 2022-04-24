
# TODO: Gen-ify this like this:
# trace, _ = generate(model)
# prior = "extract parameters"(trace)
# data = "extract data not parameters"(trace) # this is a choicemap

# for t = 1:numTrials
#     inferred_params, trace = posterior(model, data)
# end


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

@gen function model(X)
    theta = zeros(2) # linear model
    theta[1] = @trace(prior(:intercept))
    theta[2] = @trace(prior(:slope))
    n = size(X, 1)
    @trace(mvnormal(X .* theta[2] .+ theta[1], LinearAlgebra.I(n)), :Y)
end

function posterior(X, Y, numSamples) # numSamples=nOuter
    obs = choicemap()
    obs[:Y] = Y
    (trace, _) = generate(model, (X,), obs)
    samples = []
    for i in 1:numSamples
        trace, _ = mh(trace, proposal, (:intercept,))
        trace, _ = mh(trace, proposal, (:slope,))
        push!(samples, get_choices(trace))
    end
    samples, trace
end

function flattenPosteriorSamples(posteriorSamples)
    function getNumParams(sample)
        count = 0
        for (addr, val) in get_values_shallow(sample)
            count += length(val)
        end
        return count
    end
    numSamples = length(posteriorSamples)
    numParams = getNumParams(posteriorSamples[1])
    # println("flattening into $numSamples by $numParams")
    samples = zeros(numSamples, numParams)
    for s in 1:numSamples
        paramNum = 1
        for (addr, val) in get_values_shallow(posteriorSamples[s])
            # println("pNum $paramNum end $(paramNum+length(val)-1) $val")
            samples[s, paramNum:paramNum+length(val)-1] .= val
            paramNum += length(val)
        end
    end
    return samples
end

function thetaQuantile(dist, theta)
    dist = sort(vec(dist))
    if theta < dist[1]
        return 0
    end

    for i = 1:length(dist)-1
        if (dist[i] < theta && theta <= dist[i+1])
            return i
        end
    end
    return dist[end]
end

function isApproxUniform(dist, numTrials, numSamples, confidence=0.05)
    numParams = size(dist, 2)
    @test numTrials == size(dist, 1)

    for h in 1:numParams
        d = Distributions.Uniform(0, numSamples)
        # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        T = HypothesisTests.ApproximateOneSampleKSTest(dist[:, h], d)
        p = HypothesisTests.pvalue(T)
        # https://en.wikipedia.org/wiki/Bonferroni_correction#Definition
        if p > (confidence / numParams) # Bonferroni correction
            return false
        end
    end

    return true # reject null hypothesis
end

# """
# Assumes outcome has symbol `:Y`
# """
# function simulationBasedCalibration(numSamples, model, hyperparams, X, T)
#     numTrials = 100 * numSamples
#     obs = choicemap()
#     # get num params
#     initial_trace, _ = generate(model, (X, T,), obs)
#     initial_choices = get_choices(initial_trace)
#     outcome_selection = select(:Y)
#     params_selection = complement(outcome_selection)
#     true_params = get_selected(initial_choices, params_selection)
#     theta = flattenPosteriorSamples([true_params])
#     numParams = length(get_values_shallow(true_params))

# end

@testset "Simple gen model SBC" begin
    numSamples = 10
    numTrials = 100 * numSamples
    X = collect(1:50)
    obs = choicemap()

    # get num params
    initial_trace, _ = generate(model, (X,), obs)
    initial_choices = get_choices(initial_trace)
    outcome_selection = select(:Y)
    params_selection = complement(outcome_selection)
    true_params = get_selected(initial_choices, params_selection)
    theta = flattenPosteriorSamples([true_params])
    numParams = length(get_values_shallow(true_params))

    # setup quantile sampling
    quantileSamples = zeros(numTrials, numParams)

    for t = 1:numTrials
        initial_trace, _ = generate(model, (X,), obs)
        initial_choices = get_choices(initial_trace)

        outcome_selection = select(:Y)
        outcome_choices = get_selected(initial_choices, outcome_selection)
        Y = outcome_choices[:Y]
        # println("Y $Y")

        params_selection = complement(outcome_selection)
        true_params = get_selected(initial_choices, params_selection)
        # println("params $true_params")
        theta = flattenPosteriorSamples([true_params])
        # println("theta $theta")
        numParams = length(get_values_shallow(true_params))
        # println("numParams $numParams")

        posteriorSamples, trace = posterior(X, Y, numSamples)

        # println("flattening posterior samples")
        samples = flattenPosteriorSamples(posteriorSamples)

        for h = 1:numParams
            quantileSamples[t, h] = thetaQuantile(samples[:, h], theta[h])
        end
    end
    # check if quantiles are uniformly distributed
    if isApproxUniform(quantileSamples, numTrials, numSamples)
        return true
    end
    false
end