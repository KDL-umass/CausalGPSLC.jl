function simulationBasedCalibration(prior, likelihood, posterior, numTrials, numSamples; confidence=0.05)
    theta = prior() # draw "true" theta from prior 
    numParams = size(theta, 1) # theta is 1D
    quantileSamples = zeros((numTrials, numParams))

    for t = 1:numTrials # N in columbia article

        theta = prior() # draw "true" theta from prior 
        y = likelihood(theta) # draw "true" y from data model

        # posterior returns array of samples (e.g. every 100th MH step)
        thetaSamples = posterior(y, numSamples) # numSamples x numParams
        @assert size(thetaSamples) == (numSamples, numParams)

        for h = 1:numParams
            # where does this true theta component fall in the sampled distribution?
            quantileSamples[t, h] = thetaQuantile(thetaSamples[:, h], theta[h])
        end
    end
    # check if quantiles are uniformly distributed
    if isApproxUniform(quantileSamples, numTrials, numSamples, confidence)
        return true
    end
    false
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

function isApproxUniform(dist, numTrials, numSamples, confidence)
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

@testset "Simulation-Based Calibration" begin
    # https://statmodeling.stat.columbia.edu/2021/09/03/simulation-based-calibration-some-challenges-and-directions-for-future-research/

    @testset "Verify SBC procedure with prior" begin
        numSamples = 100
        numTrials = 100 * numSamples
        @gen prior() = [randn()]
        numParams = size(prior(), 1)
        @gen likelihood(theta) = [randn() * theta]
        @gen posterior(y, numSamples) = reshape([prior()[1] for i = 1:numSamples], (numSamples, numParams))

        @test simulationBasedCalibration(prior, likelihood, posterior, numTrials, numSamples)
    end

    @testset "GPSLC" begin
        # test the inference algorithms
        hyperparams = getHyperParameters()
        @gen prior() = Dict{String,Any}(
            "tyLS" => lengthscaleFromPriorNoUNoX(hyperparams),
            "yNoise" => sampleNoiseFromPrior(hyperparams)[3],
            "yScale" => @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale),
        )
        @gen likelihood(theta) = @trace(generateY(nothing, nothing, T, theta["tyLS"], theta["yScale"], theta["yNoise"]))
    end

    # I'm trying to map the existing GPSLC functions onto the SBC setup, but I'm having trouble... It's all very nested together I think. I'm starting simple with just binary T and no U or X:
    #  What I have so far is that the prior for what we're doing inference over:
    #     yNoise, tyLS, and yScale seem to be the parameters we're doing inference over
    #  Then, we need to be able to sample Y from the parameters we're doing inference over, which is the NoCovNoUBinaryGPSLC function you had defined (might have been named slightly differently)
    #     Y comes from a mvnormal on a ycov from tyCovLog, yScale, and yNoise
    #  Lastly, we need to do posterior inference on the parameters yNoise, tyLS, and yScale from this sampled Y
    #     So we get yNoise, tyLS, and yScale from the Posterior with the right inputs. 

end