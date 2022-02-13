module Estimation

using LinearAlgebra
using Gen

include("kernel.jl")
using .Kernel

export conditionalITE, conditionalSATE, ITEsamples, SATEsamples

"""Full Model Continuous/Binary"""
function conditionalITE(uyLS::Array{Float64,1}, tyLS::Float64, xyLS::Array{Float64},
    yNoise::Float64, yScale::Float64, U::Array,
    X::Array, T, Y::Array, doT)

    nU = length(U)
    nX = length(X)
    n = length(T)

    # e.g. U's contribution to log cov mat of Y
    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
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
function conditionalITE(uyLS::Array{Float64,1}, tyLS::Float64, xyLS::Nothing,
    yNoise::Float64, yScale::Float64, U::Array,
    X::Nothing, T, Y::Array, doT)

    nU = length(U)
    n = length(T)

    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(uyCovLog + tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(uyCovLog + tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(uyCovLog + tyCovLogSS, yScale, 0.0)
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
function conditionalITE(uyLS::Nothing, tyLS::Float64, xyLS::Array{Float64},
    yNoise::Float64, yScale::Float64, U::Nothing,
    X::Array, T, Y::Array, doT)

    nX = length(X)
    n = length(T)

    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
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
function conditionalITE(uyLS::Nothing, tyLS::Float64, xyLS::Nothing,
    yNoise::Float64, yScale::Float64, U::Nothing,
    X::Nothing, T, Y::Array, doT)

    n = length(T)

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

"""Conditional Sample Average Treatment Effect"""
function conditionalSATE(uyLS, tyLS::Float64, xyLS,
    yNoise::Float64, yScale::Float64, U,
    X, T, Y::Array{Float64}, doT)

    MeanITE, CovITE = conditionalITE(uyLS, tyLS, xyLS, yNoise, yScale, U, X, T, Y, doT)

    MeanSATE = sum(MeanITE) / length(T)
    VarSATE = sum(CovITE) / length(T)^2
    return MeanSATE, VarSATE
end

"""Individual Treatment Effect Samples"""
function ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
    nMixtures = length(MeanITEs[:, 1])
    n = length(MeanITEs[1, :])

    samples = zeros(nMixtures * nSamplesPerMixture, n)
    i = 0
    for j in 1:nMixtures
        mean = MeanITEs[j]
        cov = CovITEs[j]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[i, :] = mvnormal(mean, cov)
        end
    end
    return samples
end

"""Sample Average Treatment Effect samples"""
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

end
