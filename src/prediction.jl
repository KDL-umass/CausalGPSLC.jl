function counterfactualOutcomeDistribution(
    uyLS::Union{Vector{Float64},Nothing},
    xyLS::Union{Array{Float64},Nothing}, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Union{Confounders,Nothing}, X::Union{Covariates,Nothing},
    T::Treatment, Y::Outcome, doT::Intervention
)
    Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22 = likelihoodDistribution(
        uyLS, xyLS, tyLS, yNoise, yScale, U, X, T, Y, doT
    )

    MeanY = CovWW \ CovWW * Y
    MeanYs = CovWWs \ CovWW * Y
    MeanYYs = vcat(MeanY, MeanYs)

    CovYYs = hcat(vcat(CovC11, CovC21), vcat(CovC12, CovC22))
    CovYYs = hcat(vcat(CovC11, CovC21), vcat(CovC12, CovC22))
    return MeanYYs, CovYYs
end

function counterfactualOutcomeDistribution(g::GPSLCObject, posteriorSampleIdx::Int64, doT::Intervention)
    uyLS, xyLS, tyLS, yNoise, yScale, U = extractParameters(g, posteriorSampleIdx)
    counterfactualOutcomeDistribution(uyLS, xyLS, tyLS, yNoise, yScale, U, g.X, g.T, g.Y, doT)
end

function predictCounterfactualOutcomes(g::GPSLCObject, doT::Intervention, nSamplesPerMixture::Int64)
    n = getN(g)
    nps = getNumPosteriorSamples(g)
    Means = zeros(nps, 2 * n)
    Covs = zeros(nps, 2 * n, 2 * n)
    idx = 1
    for i in @mock tqdm(burnIn:stepSize:nOuter)
        MeanYYs, CovYYs = counterfactualOutcomeDistribution(g, i, doT)

        Means[idx, :] = MeanYYs
        Covs[idx, :, :] = LinearAlgebra.Symmetric(CovYYs) +
                          I * g.hyperparams.predictionCovarianceNoise

        idx += 1
    end

    nMixtures, n = size(Means)
    samples = zeros(n, nMixtures * nSamplesPerMixture)
    i = 0
    for j in 1:nMixtures
        mean = Means[j, :]
        cov = Covs[j, :, :]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[:, i] = mvnormal(mean, cov)[-n:end]
        end
    end
    return samples
end