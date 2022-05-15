"""
Reshape `posteriorSamples`, a Vector of `DSLChoiceMap`s 
into matrix samples by parameters.
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

"""Calculate the quantile of `dist` that this `theta` falls into."""
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

"""Confirm that dist is approximately uniform with `confidence`."""
function isApproxUniform(dist, numTrials, numSamples, confidence=0.05)
    numParams = size(dist, 2)
    @test numTrials == size(dist, 1)
    smallNoise = rand(size(dist)...)
    # ties can come from getting stuck in random walk
    # could add super small amount of jitter to samples to remove
    dist .+= smallNoise / 1e-20
    @assert length(dist) == length(unique(dist)) "KS-test requires unique items"

    for h in 1:numParams
        d = Distributions.Uniform(0, numSamples)

        # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        hypothesisTest = HypothesisTests.ApproximateOneSampleKSTest(dist[:, h], d)

        p = HypothesisTests.pvalue(hypothesisTest)

        # https://en.wikipedia.org/wiki/Bonferroni_correction#Definition
        if p > (confidence / numParams) # Bonferroni correction
            println("SBC Failed with p $p > $(confidence / numParams)")
            return false # accept null hypothesis
        end
    end

    return true # reject null hypothesis
end

"""Assumes outcome has symbol `:Y`."""
function simulationBasedCalibration(model, posterior,
    priorparams, n, nU, nX, nOuter, nMHInner, nESInner
    ; numTrials=nOuter * 25)

    numSamples = nOuter

    # Get total number of parameters in the model
    initial_trace, _ = generate(model, (priorparams, n, nU, nX))
    initial_choices = Gen.get_choices(initial_trace)
    data_selection = Gen.select(:X, :T, :Y)
    params_selection = Gen.complement(data_selection)
    true_params = Gen.get_selected(initial_choices, params_selection)
    theta = flattenPosteriorSamples([true_params])
    # goal:
    numParams = length(get_values_shallow(true_params))
    # end getting parameters

    # setup quantile sampling
    quantileSamples = zeros(numTrials, numParams)

    for t = 1:numTrials
        initial_trace, _ = generate(model, (priorparams, n, nU, nX,))
        initial_choices = Gen.get_choices(initial_trace)
        data_selection = Gen.select(:X, :T, :Y)
        data_choices = Gen.get_selected(initial_choices, data_selection)

        if Gen.has_value(data_choices, :X => 1 => :X)
            X = zeros(n, nX)
            for k in 1:nX
                X[:, k] = data_choices[:X=>k=>:X]
            end
        else
            X = nothing
        end
        if Gen.has_value(data_choices, :T)
            T = data_choices[:T]
        elseif Gen.has_value(data_choices, :T => 1 => :T)
            T = zeros(n)
            for i in 1:n
                T[i] = data_choices[:T=>i=>:T]
            end
        end

        Y = data_choices[:Y]

        params_selection = Gen.complement(data_selection)
        true_params = Gen.get_selected(initial_choices, params_selection)
        theta = flattenPosteriorSamples([true_params])

        posteriorSamples, trace = posterior(priorparams, X, T, Y, nU, nOuter,
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


@testset "Simulation-based Calibration" begin
    priorparams, n, nU, nX, X, binaryT, realT = getToyData(10)
    nOuter = numSamples = 5
    nMHInner = 2
    nESInner = 2

    @testset "Simple gen model SBC" begin
        model, posterior = getToyModel()
        @test simulationBasedCalibration(model, posterior, priorparams, n, nothing, nothing, 10, nothing, nothing)
    end

    @testset "Binary Treatment, No U, No Cov" begin
        @test simulationBasedCalibration(
            GPSLCNoUNoCovBinaryT, Posterior, priorparams,
            n, nothing, nothing, numSamples, nothing, nothing)
    end

    @testset "Binary Treatment, No Cov" begin
        @test simulationBasedCalibration(
            GPSLCNoCovBinaryT, Posterior, priorparams,
            n, nU, nothing, numSamples, nMHInner, nESInner)
    end

    @testset "Binary Treatment, No U" begin
        @test simulationBasedCalibration(
            GPSLCNoUBinaryT, Posterior, priorparams,
            n, nothing, nX, numSamples, nMHInner, nESInner)
    end

    @testset "Binary Treatment, Full Model" begin
        @test simulationBasedCalibration(
            GPSLCBinaryT, Posterior, priorparams,
            n, nU, nX, numSamples, nMHInner, nESInner)
    end

    @testset "Continous Treatment, No U, No Cov" begin
        @test simulationBasedCalibration(
            GPSLCNoUNoCovRealT, Posterior, priorparams,
            n, nothing, nothing, numSamples, nothing, nothing)
    end

    @testset "Continous Treatment, No Cov" begin
        @test simulationBasedCalibration(
            GPSLCNoCovRealT, Posterior, priorparams,
            n, nU, nothing, numSamples, nMHInner, nESInner)
    end

    @testset "Continous Treatment, No U" begin
        @test simulationBasedCalibration(
            GPSLCNoURealT, Posterior, priorparams,
            n, nothing, nX, numSamples, nMHInner, nESInner)
    end

    @testset "Continous Treatment, Full Model" begin
        @test simulationBasedCalibration(
            GPSLCRealT, Posterior, priorparams,
            n, nU, nX, numSamples, nMHInner, nESInner)
    end
end
