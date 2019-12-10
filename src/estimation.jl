module Estimation

# +
using LinearAlgebra
using Gen

include("model.jl")
using .Model

import Base.show
export conditionalSATE, SATE, SATEsamples

# +
# TODO: Test and modify for new API

# function conditionalSATE(uyLS::Float64, xyLS::Float64, epsyLS::Float64, U, epsY, X, Y, doX)
# #   Generate a new post-intervention instance for each data instance in
# #   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
# #   with doX.
    
# #   This assumes that the confounder U, the exogenous noise, and kernel hyperparameters are known. 
# #   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
# #   be used to compute monte carlo estimates.
    
#     n = length(U)
    
#     CovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, X, reshape(X, 1, n), xyLS, epsY, reshape(epsY, 1, n), epsyLS)
#     CovY = Symmetric(CovY)
        
# #   k_Y,Y_x in the overleaf doc. The cross covariance block is not in general symettric.
#     crossCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, reshape(X, 1, n), xyLS, epsY, reshape(epsY, 1, n), epsyLS)

# #   k_Y_x in the overleaf doc.
#     intCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, doX, xyLS, epsY, reshape(epsY, 1, n), epsyLS)
#     intCovY = Symmetric(intCovY)
    
#     condMean = crossCovY * (CovY \ Y)
#     condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
#     effectMean = sum(condMean-Y)/n
#     effectVar = sum(condCov)/n
    
#     return effectMean, effectVar
# end
# -

# Overloading the function depending on whether there is an exogenous noise term.

function conditionalSATE(uyLS::Float64, xyLS::Float64, yNoise::Float64, yScale::Float64, U, X, Y, doX)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovWW = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, X, reshape(X, 1, n), xyLS, yScale)
    CovWW = Symmetric(CovWW)
    
    CovWWp = Symmetric(CovWW + (yNoise * 1I))
    
#   K(W, W_*) in the overleaf doc. The cross covariance block is not in general symettric.
    CovWWs = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, X, doX, xyLS, yScale)
    
#   K(W_*, W_*) in the overleaf doc.
    CovWsWs = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, doX, xyLS, yScale)
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
    
    MeanSATE = sum(MeanCF)/n
    VarSATE = sum(CovCF)/n
    
    return MeanITE, CovITE, MeanSATE, VarSATE
end

# +
# TODO: Test and modify for new API

# function SATE(postHyp, postU, postEps, X, Y, doX)
    
#     nPostSamples = length(postU)
    
#     effectMeans = zeros(nPostSamples)
#     effectVars = zeros(nPostSamples)
    
#     for i in 1:nPostSamples
#          effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["xyLS"], 
#                                                          postHyp[i]["epsyLS"], postU[i], 
#                                                          postEps[i], X, Y, doX)
#     end
    
#     n = length(effectMeans)
    
#     totalMean = sum(effectMeans)/n
#     totalVar = sum(effectVars)/n + sum(effectMeans .* effectMeans)/n - ((sum(effectMeans))^2)/n^2
    
#     return totalMean, totalVar
# end
# -

# Overloading the function depending on whether there is an exogenous noise term. 
# If not, use the AdditiveNoiseGPROC model.

function SATE(PosteriorSamples, X, Y, doX)
    
    n = length(PosteriorSamples)
    MeanSATEs = zeros(n)
    VarSATEs = zeros(n)
    
    for i in 1:n
        _, _, MeanSATEs[i], VarSATEs[i] = conditionalSATE(PosteriorSamples[i][:uyLS], 
                                                          PosteriorSamples[i][:xyLS], 
                                                          PosteriorSamples[i][:yNoise], 
                                                          PosteriorSamples[i][:yScale],
                                                          PosteriorSamples[i][:U], 
                                                          X, 
                                                          Y, 
                                                          doX)
    end
    
    return MeanSATEs, VarSATEs
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
