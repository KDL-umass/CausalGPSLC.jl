module Estimation

# +
using LinearAlgebra
using Gen

include("model.jl")
using .Model

import Base.show
export conditionalITE, conditionalSATE, ITE, LinearSATE, SATE, ITEsamples, SATEsamples

# +
# TODO: Test and modify for new API

# function conditionalSATE(uyLS::Float64, tyLS::Float64, epsyLS::Float64, U, epsY, T, Y, doT)
# #   Generate a new post-intervention instance for each data instance in
# #   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
# #   with doT.
    
# #   This assumes that the confounder U, the exogenous noise, and kernel hyperparameters are known. 
# #   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
# #   be used to compute monte carlo estimates.
    
#     n = length(U)
    
#     CovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, T, reshape(T, 1, n), tyLS, epsY, reshape(epsY, 1, n), epsyLS)
#     CovY = Symmetric(CovY)
        
# #   k_Y,Y_x in the overleaf doc. The cross covariance block is not in general symettric.
#     crossCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, reshape(X, 1, n), xyLS, epsY, reshape(epsY, 1, n), epsyLS)

# #   k_Y_x in the overleaf doc.
#     intCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doT, doT, xyLS, epsY, reshape(epsY, 1, n), epsyLS)
#     intCovY = Symmetric(intCovY)
    
#     condMean = crossCovY * (CovY \ Y)
#     condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
#     effectMean = sum(condMean-Y)/n
#     effectVar = sum(condCov)/n
    
#     return effectMean, effectVar
# end

# +
# RBF kernel for X, T, and Y. Additive Gaussian Noise.
# Also handles case where Xcol=nothing.

function conditionalITE(uyLS::Float64, tyLS::Float64, xyLS::Array{Float64}, yNoise::Float64, yScale::Float64, Xcol, U, T, Y, doT)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovWW = broadcast(y_kernel, U, U', uyLS, T, T', tyLS, Xcol, (xyLS,), yScale)
    CovWW = Symmetric(CovWW)
    
    CovWWp = Symmetric(CovWW + (yNoise * 1I))
    
#   K(W, W_*) in the overleaf doc. The cross covariance block is not in general symettric.
    CovWWs = broadcast(y_kernel, U, U', uyLS, T, doT, tyLS, Xcol, (xyLS,), yScale)
    
#   K(W_*, W_*) in the overleaf doc.
    CovWsWs = broadcast(y_kernel, U, U', uyLS, doT, doT, tyLS, Xcol, (xyLS,), yScale)
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

# +
# Overloading function for the case where ty relationship is linear. Additive Gaussian Noise.
# 

function conditionalITE(uyLS::Float64, xyLS::Array{Float64}, yNoise::Float64, yScale::Float64, Xcol, U, T, Y, doT)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovWW = broadcast(y_kernel, U, U', uyLS, T, T', Xcol, (xyLS,), yScale)
    CovWW = Symmetric(CovWW)
    
    CovWWp = Symmetric(CovWW + (yNoise * 1I))
    
#   K(W, W_*) in the overleaf doc. The cross covariance block is not in general symettric.
    CovWWs = broadcast(y_kernel, U, U', uyLS, T, doT, Xcol, (xyLS,), yScale)
    
#   K(W_*, W_*) in the overleaf doc.
    CovWsWs = broadcast(y_kernel, U, U', uyLS, doT, doT, Xcol, (xyLS,), yScale)
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
# -

# Overloading the function depending on whether there is an exogenous noise term. 
# If not, use the AdditiveNoiseGPROC model.

# +
# TODO: Test and modify for new API

# function SATE(postHyp, postU, postEps, T, Y, doX)
    
#     nPostSamples = length(postU)
    
#     effectMeans = zeros(nPostSamples)
#     effectVars = zeros(nPostSamples)
    
#     for i in 1:nPostSamples
#          effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["tyLS"], 
#                                                          postHyp[i]["epsyLS"], postU[i], 
#                                                          postEps[i], T, Y, doT)
#     end
    
#     n = length(effectMeans)
    
#     totalMean = sum(effectMeans)/n
#     totalVar = sum(effectVars)/n + sum(effectMeans .* effectMeans)/n - ((sum(effectMeans))^2)/n^2
    
#     return totalMean, totalVar
# end

# +
# RBF Kernel for Y = f(U, T, X) with additive gaussian noise. 
# Works with Xcol=nothing.
function conditionalSATE(uyLS::Float64, tyLS::Float64, xyLS::Array{Float64}, yNoise::Float64, yScale::Float64, 
                        Xcol, U, T, Y, doT)
    
    MeanITE, CovITE = conditionalITE(uyLS, tyLS, xyLS, yNoise, yScale, Xcol, U, T, Y, doT)
    n = length(T)
    
    MeanSATE = sum(MeanITE)/n
    VarSATE = sum(CovITE)/n
    return MeanSATE, VarSATE
end

# RBF kernel for X -> Y and U -> Y. Linear kernel for T -> Y with additive gaussian noise. 
# Works with Xcol=nothing.
function conditionalSATE(uyLS::Float64, xyLS::Array{Float64}, yNoise::Float64, yScale::Float64, 
                         Xcol, U, T, Y, doT)
    
    MeanITE, CovITE = conditionalITE(uyLS, xyLS, yNoise, yScale, Xcol, U, T, Y, doT)
    n = length(T)
    
    MeanSATE = sum(MeanITE)/n
    VarSATE = sum(CovITE)/n
    return MeanSATE, VarSATE
end
# -

function ITE(PosteriorSamples, nX::Int, Xcol, T, Y, doT)
    nSamples = length(PosteriorSamples)
    n = length(T)
    MeanITEs = zeros(nSamples, n)
    CovITEs = zeros(nSamples, n, n)
    
    for i in 1:nSamples
        
        xyLS = [PosteriorSamples[i][:xyLS => j => :LS] for j in 1:nX]
        
        MeanITEs[i, :], CovITEs[i, :, :] = conditionalITE(PosteriorSamples[i][:uyLS],
                                                          xyLS,
                                                          PosteriorSamples[i][:tyLS], 
                                                          PosteriorSamples[i][:yNoise], 
                                                          PosteriorSamples[i][:yScale],
                                                          PosteriorSamples[i][:U], 
                                                          Xcol,
                                                          T, 
                                                          Y, 
                                                          doT)
    end
    
    return MeanITEs, CovITEs
end

# +
function SATE(PosteriorSamples, nX::Int, Xcol, T, Y, doT)
    
    nSamples = length(PosteriorSamples)
    MeanSATEs = zeros(nSamples)
    VarSATEs = zeros(nSamples)
    
    for i in 1:nSamples
        
        xyLS = [PosteriorSamples[i][:xyLS => j => :LS] for j in 1:nX]
        
        MeanSATEs[i], VarSATEs[i] = conditionalSATE(PosteriorSamples[i][:uyLS],
                                                    PosteriorSamples[i][:tyLS], 
                                                    xyLS,
                                                    PosteriorSamples[i][:yNoise], 
                                                    PosteriorSamples[i][:yScale],
                                                    Xcol,
                                                    PosteriorSamples[i][:U], 
                                                    T, 
                                                    Y, 
                                                    doT)
    end
    
    return MeanSATEs, VarSATEs
end

function LinearSATE(PosteriorSamples, nX::Int, Xcol, T, Y, doT)
    
    nSamples = length(PosteriorSamples)
    MeanSATEs = zeros(nSamples)
    VarSATEs = zeros(nSamples)
    
    for i in 1:nSamples
        
        xyLS = [PosteriorSamples[i][:xyLS => j => :LS] for j in 1:nX]
        
        MeanSATEs[i], VarSATEs[i] = conditionalSATE(PosteriorSamples[i][:uyLS],
                                                    xyLS,
                                                    PosteriorSamples[i][:yNoise], 
                                                    PosteriorSamples[i][:yScale], 
                                                    Xcol,
                                                    PosteriorSamples[i][:U],
                                                    T, 
                                                    Y, 
                                                    doT)
    end
    
    return MeanSATEs, VarSATEs
end
# -

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
