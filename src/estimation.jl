export conditionalITE, conditionalSATE, ITEsamples, SATEsamples


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

    n = size(Y, 1)
    @assert size(U, 1) == n
    @assert size(X, 1) == n
    @assert size(T, 1) == n
    nU = ndims(U) == 2 ? size(U, 2) : 1
    @assert size(uyLS, 1) == nU
    nX = ndims(X) == 2 ? size(X, 2) : 1
    @assert size(xyLS, 1) == nX

    # e.g. U's contribution to log cov mat of Y
    uyCovLog = rbfKernelLog(U, U, uyLS)
    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(uyCovLog + xyCovLog + tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(uyCovLog + xyCovLog + tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

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

    n = size(Y, 1)
    @assert size(U, 1) == n
    @assert size(T, 1) == n
    nU = ndims(U) == 2 ? size(U, 2) : 1
    @assert size(uyLS, 1) == nU

    uyCovLog = rbfKernelLog(U, U, uyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(uyCovLog .+ tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(uyCovLog .+ tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(uyCovLog .+ tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

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

    n = size(Y, 1)
    @assert size(X, 1) == n
    @assert size(T, 1) == n
    nX = ndims(X) == 2 ? size(X, 2) : 1
    @assert size(xyLS, 1) == nX

    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(xyCovLog + tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(xyCovLog + tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(xyCovLog + tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

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

    n = size(Y, 1)
    @assert size(T, 1) == n

    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in paper.
    CovWsWs = processCov(tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

    MeanITE = (CovWWs' - CovWW) * (CovWWp \ Y)

    CovITE = CovC11 - CovC12 - CovC21 + CovC22

    return MeanITE, CovITE
end


"""
Wrapper for conditionalITE that extracts parameters from `g::GPSLCObject` at posterior sample `i` and applies intervention `doT`.
"""
function conditionalITE(g::GPSLCObject, i::Int64, doT::Intervention)
    n = getN(g)
    nU = getNU(g)
    uyLS = zeros(nU)
    U = zeros(n, nU)
    for u in 1:nU
        uyLS[u] = g.posteriorSamples[i][:uyLS=>u=>:LS]
        U[:, u] = g.posteriorSamples[i][:U=>u=>:U]
    end
    U = toMatrix(U, n, nU)
    @assert size(U) == (n, nU)

    if g.X === nothing
        xyLS = nothing
    else
        nX = getNX(g)
        xyLS = zeros(nX)
        for k in 1:nX
            xyLS[k] = g.posteriorSamples[i][:xyLS=>k=>:LS]
        end
    end

    conditionalITE(
        uyLS, xyLS, g.posteriorSamples[i][:tyLS],
        g.posteriorSamples[i][:yNoise], g.posteriorSamples[i][:yScale],
        U, g.X, g.T, g.Y,
        doT)
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
        CovITEs[idx, :, :] = LinearAlgebra.Symmetric(CovITE) + I * (1e-10)
        idx += 1
    end
    return MeanITEs, CovITEs
end


"""Individual Treatment Effect Samples"""
function ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
    nMixtures, n = size(MeanITEs)

    samples = zeros(nMixtures * nSamplesPerMixture, n)
    i = 0
    for j in 1:nMixtures
        mean = MeanITEs[j, :]
        cov = CovITEs[j, :, :]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[i, :] = mvnormal(mean, cov)
        end
    end
    return samples
end


"""Conditional Sample Average Treatment Effect"""
function conditionalSATE(MeanITE, CovITE)
    n = size(MeanITE, 1)
    MeanSATE = sum(MeanITE) / n
    VarSATE = sum(CovITE) / n^2
    return MeanSATE, VarSATE
end

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
Sample Average Treatment Effect samples

returns 
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