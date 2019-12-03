module Estimation

# +
using LinearAlgebra

include("model.jl")
using .Model

import Base.show
export conditionalSATE, SATE
# -

function conditionalSATE(uyLS::Float64, xyLS::Float64, epsyLS::Float64, U, epsY, X, Y, doX)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U, the exogenous noise, and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, X, reshape(X, 1, n), xyLS, epsY, reshape(epsY, 1, n), epsyLS)
    CovY = Symmetric(CovY)
        
#   k_Y,Y_x in the overleaf doc. The cross covariance block is not in general symettric.
    crossCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, reshape(X, 1, n), xyLS, epsY, reshape(epsY, 1, n), epsyLS)

#   k_Y_x in the overleaf doc.
    intCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, doX, xyLS, epsY, reshape(epsY, 1, n), epsyLS)
    intCovY = Symmetric(intCovY)
    
    condMean = crossCovY * (CovY \ Y)
    condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
    effectMean = sum(condMean-Y)/n
    effectVar = sum(condCov)/n
    
    return effectMean, effectVar
end

# Overloading the function depending on whether there is an exogenous noise term.

function conditionalSATE(uyLS::Float64, xyLS::Float64, U, X, Y, doX)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, X, reshape(X, 1, n), xyLS)
    CovY = Symmetric(CovY)
    
#   k_Y,Y_x in the overleaf doc. The cross covariance block is not in general symettric.
    crossCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, reshape(X, 1, n), xyLS)
    
#   k_Y_x in the overleaf doc.
    intCovY = broadcast(y_kernel, U, reshape(U, 1, n), uyLS, doX, doX, xyLS)
    intCovY = Symmetric(intCovY)
    
    condMean = crossCovY * (CovY \ Y)
    condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
    effectMean = sum(condMean-Y)/n
    effectVar = sum(condCov)/n
    
    return effectMean, effectVar
end

function SATE(postHyp, postU, postEps, X, Y, doX)
    
    nPostSamples = length(postU)
    
    effectMeans = zeros(nPostSamples)
    effectVars = zeros(nPostSamples)
    
    for i in 1:nPostSamples
         effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["xyLS"], 
                                                         postHyp[i]["epsyLS"], postU[i], 
                                                         postEps[i], X, Y, doX)
    end
    
    n = length(effectMeans)
    
    totalMean = sum(effectMeans)/n
    totalVar = sum(effectVars)/n + sum(effectMeans .* effectMeans)/n - ((sum(effectMeans))^2)/n^2
    
    return totalMean, totalVar
end

# Overloading the function depending on whether there is an exogenous noise term. 
# If not, use the AdditiveNoiseGPROC model.

function SATE(postHyp, postU, X, Y, doX)
    
    n = length(postU[:, 1])
    effectMeans = zeros(n)
    effectVars = zeros(n)
    
    for i in 1:n
        effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["xyLS"], postU[i, :], 
                                                         X, Y, doX)
    end
    
    totalMean = sum(effectMeans)/n
    totalVar = sum(effectVars)/n + sum(effectMeans .* effectMeans)/n - ((sum(effectMeans))^2)/n^2
    
    return totalMean, totalVar
end

end
