module Estimation

# +
using LinearAlgebra
using Gen

include("model.jl")
using .Model

import Base.show
export conditionalITE, conditionalSATE, ITE, LinearSATE, SATE, ITEsamples, SATEsamples

# +
function conditionalITE(uyLS::Array{Float64, 1}, tyLS::Float64, xyLS::Array{Float64}, 
                        yNoise::Float64, yScale::Float64, U::Array, 
                        X::Array, T, Y::Array, doT)
    
    nU = length(U)
    nX = length(X)
    n = length(T)
    
    uyCovLog   = sum(broadcast(rbfKernelLog, U, U, uyLS))
    xyCovLog   = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog   = rbfKernelLog(T, T, tyLS)
    tyCovLogS  = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)
    
    CovWW  = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, 0.)
    CovWW  = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))
    
    #   K(W, W_*) in the overleaf doc. The cross covariance block is not in general symettric.
    CovWWs = processCov(uyCovLog + xyCovLog + tyCovLogS, yScale, 0.)
    
#   K(W_*, W_*) in the overleaf doc.
    CovWsWs = processCov(uyCovLog + xyCovLog + tyCovLogSS, yScale, 0.)
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

function conditionalSATE(uyLS::Array{Float64, 1}, tyLS::Float64, xyLS::Array{Float64}, 
                        yNoise::Float64, yScale::Float64, U::Array, 
                        X::Array, T, Y::Array{Float64}, doT)
    
    MeanITE, CovITE = conditionalITE(uyLS, tyLS, xyLS, yNoise, yScale, U, X, T, Y, doT)
    
    MeanSATE = sum(MeanITE)/length(T)
    VarSATE = sum(CovITE)/length(T)^2
    return MeanSATE, VarSATE
end

function ITE(PosteriorSamples, nX::Int, Xcol, T, Y, doT)
    nSamples = length(PosteriorSamples)
    n = length(T)
    MeanITEs = zeros(nSamples, n)
    CovITEs = zeros(nSamples, n, n)
    
    for i in 1:nSamples
        
        post = PosteriorSamples[i]
        
        xyLS = [PosteriorSamples[i][:xyLS => j => :LS] for j in 1:nX]
        uyLS = [post[:uyLS => i => :LS] for i in 1:nU]
        tyLS = post[:tyLS]
        xyLS = [post[:xyLS => i => :LS] for i in 1:nX]
        yNoise = post[:yNoise]
        yScale = post[:yScale]
        U = [post[:U => i => :U] for i in 1:nU]
        
        MeanITEs[i, :], CovITEs[i, :, :] = conditionalITE(uyLS, tyLS, xyLS, yNoise, yScale, U, X, T, Y, doT)
    end
    
    return MeanITEs, CovITEs
end

function SATE(PosteriorSamples, nX::Int, Xcol, T, Y, doT)
    
    nSamples = length(PosteriorSamples)
    MeanSATEs = zeros(nSamples)
    VarSATEs = zeros(nSamples)
    
    for i in 1:nSamples
        
        post = PosteriorSamples[i]
        
        xyLS = [PosteriorSamples[i][:xyLS => j => :LS] for j in 1:nX]
        uyLS = [post[:uyLS => i => :LS] for i in 1:nU]
        tyLS = post[:tyLS]
        xyLS = [post[:xyLS => i => :LS] for i in 1:nX]
        yNoise = post[:yNoise]
        yScale = post[:yScale]
        U = [post[:U => i => :U] for i in 1:nU]
        
        MeanSATEs[i], VarSATEs[i] = conditionalSATE(uyLS, tyLS, xyLS, yNoise, yScale, U, X, T, Y, doT)
    end
    
    return MeanSATEs, VarSATEs
end

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
