@gen function simulationBasedCalibration(prior, likelihood, posterior, numTrials, numSamples)
    theta = @trace(prior()) # draw "true" theta from prior 
    numParams = size(theta, 1) # theta is 1D

    quantileSamples = zeros((numTrials, numParams))

    for t = 1:numTrials # N in columbia article

        theta = @trace(prior()) # draw "true" theta from prior 
        y = @trace(likelihood(theta)) # draw "true" y from data model

        thetaSamples = zeros(numSamples, numParams)
        for s = 1:numSamples # L in columbia article
            # draw inference samples from posterior
            thetaSamples[s, :] = @trace(posterior(y))
        end

        for h = 1:numParams
            # where does this true theta component fall in the sampled distribution?
            quantileSamples[t, h] = thetaQuantile(thetaSamples[:, h], theta[h])
        end
    end
    # return quantileSamples
    # check if quantiles are uniformly distributed
    if isApproxUniform(quantileSamples, numTrials, numSamples)
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

function isApproxUniform(dist, numTrials, numSamples)
    # https://en.wikipedia.org/wiki/Bonferroni_correction#Definition
    # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    dist = vec(dist)
    d = Distributions.Uniform(0, numSamples)
    T = HypothesisTests.ApproximateOneSampleKSTest(dist, d)
    p = HypothesisTests.pvalue(T)
    return p < 0.05
end

@testset "Simulation-Based Calibration" begin
    # SBC: Run univariate sbc for each dimension
    # Apply Bonferroni correction to adjust p-value (multiple hypothesis correction)
    # https://statmodeling.stat.columbia.edu/2021/09/03/simulation-based-calibration-some-challenges-and-directions-for-future-research/

    @testset "Verify SBC procedure with prior" begin
        numSamples = 100
        numTrials = 100 * numSamples
        @gen prior() = [randn()]
        @gen likelihood(theta) = [randn() * theta]
        @gen posterior(y) = prior()

        @test simulationBasedCalibration(prior, likelihood, posterior, numTrials, numSamples)
    end

end