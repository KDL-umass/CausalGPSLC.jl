import FunctionalCollections

export CausalGPSLCRealT,
    CausalGPSLCNoURealT, CausalGPSLCNoCovRealT, CausalGPSLCNoUNoCovRealT,
    CausalGPSLCBinaryT,
    CausalGPSLCNoUBinaryT, CausalGPSLCNoCovBinaryT, CausalGPSLCNoUNoCovBinaryT

# Real Valued Treatment

"""Continous CausalGPSLC, with Latent Confounders (U) and Covariates (X)"""
@gen function CausalGPSLCRealT(priorparams::PriorParameters, n, nU, nX)::Outcome
    uNoise = @trace(sampleNoiseFromPriorU(priorparams))
    xNoise = @trace(sampleNoiseFromPriorX(priorparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorUX(priorparams, nU, nX))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(priorparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    xScale = @trace(scaleFromPriorX(priorparams, nX))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    U = @trace(generateUfromSigmaU(priorparams["SigmaU"], uNoise, n, nU))
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateRealTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end

"""No Latent Confounders (no U), Continuous CausalGPSLC"""
@gen function CausalGPSLCNoURealT(priorparams::PriorParameters, n::Int64, nU::Nothing, nX::Int64)::Outcome
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(priorparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    X = @trace(generateXfromPrior(priorparams, n, nX))
    T = @trace(generateRealTfromX(nothing, X, nothing, xtLS, tScale, tNoise))
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), Continuous CausalGPSLC"""
@gen function CausalGPSLCNoCovRealT(priorparams::PriorParameters, n::Int64, nU::Int64, nX::Nothing)::Outcome
    uNoise = @trace(sampleNoiseFromPriorU(priorparams))
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    utLS, uyLS = @trace(lengthscaleFromPriorU(priorparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    U = @trace(generateUfromSigmaU(priorparams["SigmaU"], uNoise, n, nU))
    T = @trace(generateRealTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), No Latent Confounders (no U), Continuous CausalGPSLC"""
@gen function CausalGPSLCNoUNoCovRealT(priorparams::PriorParameters, n::Int64, nU::Nothing, nX::Nothing)::Outcome
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    T = @trace(generateRealTfromPrior(priorparams, n))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end

# Binary Valued Treatment

"""Binary Treatment CausalGPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function CausalGPSLCBinaryT(priorparams::PriorParameters, n::Int64, nU::Int64, nX::Int64)::Outcome
    uNoise = @trace(sampleNoiseFromPriorU(priorparams))
    xScale = @trace(scaleFromPriorX(priorparams, nX))
    xNoise = @trace(sampleNoiseFromPriorX(priorparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorUX(priorparams, nU, nX))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(priorparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    U = @trace(generateUfromSigmaU(priorparams["SigmaU"], uNoise, n, nU))
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateBinaryTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), Binary Treatment CausalGPSLC"""
@gen function CausalGPSLCNoCovBinaryT(priorparams::PriorParameters, n::Int64, nU::Int64, nX::Nothing)::Outcome
    n = size(priorparams["SigmaU"])[1]
    uNoise = @trace(sampleNoiseFromPriorU(priorparams))
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    utLS, uyLS = @trace(lengthscaleFromPriorU(priorparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    U = @trace(generateUfromSigmaU(priorparams["SigmaU"], uNoise, n, nU))
    T = @trace(generateBinaryTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))
    return Y
end

"""No latent confounders (no U), Binary Treatment CausalGPSLC"""
@gen function CausalGPSLCNoUBinaryT(priorparams::PriorParameters, n::Int64, nU::Nothing, nX::Int64)::Outcome
    tNoise = @trace(sampleNoiseFromPriorT(priorparams))
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(priorparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    tScale = @trace(scaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    X = @trace(generateXfromPrior(priorparams, n, nX))
    T = @trace(generateBinaryTfromX(nothing, X, nothing, xtLS, tScale, tNoise))
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))
    return Y
end

"""No covariates (no X), no latent confounders (no U) for Binary Treatment CausalGPSLC"""
@gen function CausalGPSLCNoUNoCovBinaryT(priorparams::PriorParameters, n::Int64, nU::Nothing, nX::Nothing)::Outcome
    yNoise = @trace(sampleNoiseFromPriorY(priorparams))
    tyLS = @trace(lengthscaleFromPriorT(priorparams))
    yScale = @trace(scaleFromPriorY(priorparams))
    T = @trace(generateBinaryTfromPrior(priorparams, n))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end
