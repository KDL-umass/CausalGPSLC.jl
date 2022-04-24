"""
Reshape `posteriorSamples`, a Vector of `DSLChoiceMap`s 
into matrix samples by parameters
"""
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

"""Calculate the quantile of `dist` that this `theta` falls into"""
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

"""Confirm that dist is approximately uniform with `confidence`"""
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

"""
Assumes outcome has symbol `:Y`
"""
function simulationBasedCalibration(model, posterior,
    hyperparams, nU, X, T, nOuter, nMHInner, nESInner; numTrials=nOuter * 100)

    numSamples = nOuter
    # get total number of parameters in the model
    obs = choicemap()
    initial_trace, _ = generate(model, (hyperparams, nU, X, T,), obs)
    initial_choices = Gen.get_choices(initial_trace)
    outcome_selection = Gen.select(:Y)
    params_selection = Gen.complement(outcome_selection)
    true_params = Gen.get_selected(initial_choices, params_selection)
    theta = flattenPosteriorSamples([true_params])
    numParams = length(get_values_shallow(true_params))

    # setup quantile sampling
    quantileSamples = zeros(numTrials, numParams)

    for t = 1:numTrials
        initial_trace, _ = generate(model, (hyperparams, nU, X, T,), obs)
        initial_choices = get_choices(initial_trace)
        outcome_selection = Gen.select(:Y)
        outcome_choices = get_selected(initial_choices, outcome_selection)
        Y = outcome_choices[:Y]

        params_selection = complement(outcome_selection)
        true_params = get_selected(initial_choices, params_selection)
        theta = flattenPosteriorSamples([true_params])
        numParams = length(get_values_shallow(true_params))

        posteriorSamples, trace = posterior(hyperparams, X, T, Y, nU, nOuter,
            nMHInner, nESInner) # numSamples=nOuter

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

@testset "Simple gen model SBC" begin
    X = collect(1:50)
    T = nothing
    model, posterior = getToyModel()
    @test simulationBasedCalibration(model, posterior, nothing, nothing, X, T, 10, nothing, nothing)
end

@testset "GPSLC SBC" begin
    hyperparams, n, nU, nX, X, binaryT, realT = getToyData(10)
    numSamples = 5

    @testset "Binary Treatment, No U, No Cov" begin
        @test simulationBasedCalibration(
            GPSLCNoUNoCovBinaryT, Posterior, hyperparams,
            nothing, nothing, binaryT, numSamples, nothing, nothing)
    end
end
