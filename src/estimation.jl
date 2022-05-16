export conditionalITE,
    ITEDistributions,
    ITEsamples,
    conditionalSATE,
    SATEDistributions,
    SATEsamples

"""
    Conditional Individual Treatment Estimation

`conditionalITE` takes in parameters (presumably from posterior inference)
as well as the observed and inferred data to produce individual treatment 
effects.

Params:
- `uyLS`: (optional) Kernel lengthscale for latent confounders to outcome
- `xyLS`: (optional) Kernel lengthscale for covariates to outcome
- `tyLS`: Kernel lengthscale for treatment to outcome
- `yNoise`: Gaussian noise for outcome prediction
- `yScale`: Gaussian scale for outcome prediction
- `U`: (optional) Latent confounders
- `X`: (optional) Covariates
- `T`: Treatment
- `Y`: Outcome
- `doT`: Treatment intervention

Full Model Continuous/Binary
"""
function conditionalITE(
    uyLS::Vector{Float64}, xyLS::Array{Float64}, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Confounders, X::Covariates, T::Treatment,
    Y::Outcome, doT::Intervention)

    Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22 = likelihoodDistribution(
        uyLS, xyLS, tyLS, yNoise, yScale, U, X, T, Y, doT
    )

    MeanITE = (CovWWs' - CovWW) * (CovWWp \ Y)
    CovITE = CovC11 - CovC12 - CovC21 + CovC22

    return MeanITE, CovITE
end

"""No Covariates Continuous/Binary"""
function conditionalITE(
    uyLS::Vector{Float64}, xyLS::Nothing, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Confounders, X::Nothing, T::Treatment,
    Y::Outcome, doT::Intervention)

    Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22 = likelihoodDistribution(
        uyLS, xyLS, tyLS, yNoise, yScale, U, X, T, Y, doT
    )

    MeanITE = (CovWWs' - CovWW) * (CovWWp \ Y)
    CovITE = CovC11 - CovC12 - CovC21 + CovC22

    return MeanITE, CovITE
end

"""No Confounders Continuous/Binary"""
function conditionalITE(
    uyLS::Nothing, xyLS::Vector{Float64}, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Nothing, X::Covariates, T::Treatment,
    Y::Outcome, doT::Intervention)

    Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22 = likelihoodDistribution(
        uyLS, xyLS, tyLS, yNoise, yScale, U, X, T, Y, doT
    )

    MeanITE = (CovWWs' - CovWW) * (CovWWp \ Y)
    CovITE = CovC11 - CovC12 - CovC21 + CovC22

    return MeanITE, CovITE
end

"""No Confounders No Covariates Continuous/Binary"""
function conditionalITE(
    uyLS::Nothing, xyLS::Nothing, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Nothing, X::Nothing, T::Treatment,
    Y::Outcome, doT::Intervention)

    Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22 = likelihoodDistribution(
        uyLS, xyLS, tyLS, yNoise, yScale, U, X, T, Y, doT
    )

    MeanITE = (CovWWs' - CovWW) * (CovWWp \ Y)
    CovITE = CovC11 - CovC12 - CovC21 + CovC22

    return MeanITE, CovITE
end


"""
Wrapper for conditionalITE that extracts parameters from `g::GPSLCObject` at posterior sample `i` and applies intervention `doT`.
"""
function conditionalITE(g::GPSLCObject, i::Int64, doT::Intervention)
    uyLS, xyLS, tyLS, yNoise, yScale, U = extractParameters(g, i)
    conditionalITE(uyLS, xyLS, tyLS, yNoise, yScale, U, g.X, g.T, g.Y, doT)
end

"""
    ITEDistributions

Collect MeanITEs and CovITEs from the posterior with conditionalITE.
"""
function ITEDistributions(g::GPSLCObject, doT::Intervention)
    n = getN(g)
    burnIn = g.hyperparams.nBurnIn
    stepSize = g.hyperparams.stepSize
    nOuter = g.hyperparams.nOuter

    numPosteriorSamples = length(burnIn:stepSize:nOuter)

    MeanITEs = zeros(numPosteriorSamples, n)
    CovITEs = zeros(numPosteriorSamples, n, n)

    idx = 1
    for i in @mock tqdm(burnIn:stepSize:nOuter)
        MeanITE, CovITE = conditionalITE(g, i, doT)

        MeanITEs[idx, :] = MeanITE
        CovITEs[idx, :, :] = LinearAlgebra.Symmetric(CovITE) + I * g.hyperparams.predictionCovarianceNoise
        idx += 1
    end
    return MeanITEs, CovITEs
end


"""
    Individual Treatment Effect Samples
Returns `nMixtures * nSamplesPerMixture` outcome (Y) samples for each individual
`[nMixtures * nSamplesPerMixture, n]`
where nMixtures is the number of posterior samples (nOuter)
"""
function ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
    nMixtures, n = size(MeanITEs)

    samples = zeros(n, nMixtures * nSamplesPerMixture)
    i = 0
    for j in 1:nMixtures
        mean = MeanITEs[j, :]
        cov = CovITEs[j, :, :]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[:, i] = mvnormal(mean, cov)
        end
    end
    return samples
end


"""
    conditionalSATE
Conditional Sample Average Treatment Effect
"""
function conditionalSATE(MeanITE, CovITE)
    n = size(MeanITE, 1)
    MeanSATE = sum(MeanITE) / n
    VarSATE = sum(CovITE) / n^2
    return MeanSATE, VarSATE
end

"""
    SATEDistributions
Collect SATE Mean and Variance corresponding to each posterior sample.
"""
function SATEDistributions(g::GPSLCObject, doT::Intervention)
    MeanITEs, CovITEs = ITEDistributions(g, doT)

    nMixtures, n = size(MeanITEs)
    MeanSATEs = zeros(nMixtures)
    VarSATEs = zeros(nMixtures)
    for i in 1:nMixtures
        MeanITE = MeanITEs[i, :]
        CovITE = CovITEs[i, :, :]

        MeanSATEs[i], VarSATEs[i] = conditionalSATE(MeanITE, CovITE)
    end
    return MeanSATEs, VarSATEs
end

"""
    SATEsamples
Collect Sample Average Treatment Effect corresponding to each posterior sample.

Returns a vector of `nSamplesPerMixture` samples for each posterior 
sample's SATE distribution parameters.
"""
function SATEsamples(MeanSATEs, VarSATEs, nSamplesPerMixture)
    nMixtures = length(MeanSATEs)

    samples = zeros(nMixtures * nSamplesPerMixture)

    i = 0
    for j in 1:nMixtures
        mean = MeanSATEs[j]
        var = VarSATEs[j]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[i] = normal(mean, var)
        end
    end
    return samples
end