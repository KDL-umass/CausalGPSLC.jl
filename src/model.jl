import FunctionalCollections

export GPSLCRealT,
    GPSLCNoURealT, GPSLCNoCovRealT, GPSLCNoUNoCovRealT,
    GPSLCBinaryT,
    GPSLCNoUBinaryT, GPSLCNoCovBinaryT, GPSLCNoUNoCovBinaryT

# Real Valued Treatment

"""Continous GPSLC, with Latent Confounders (U) and Covariates (X)"""
@gen function GPSLCRealT(hyperparams::HyperParameters, nU::Int64, X::Covariates, T::Treatment)::Outcome
    n = size(hyperparams["SigmaU"], 1)
    nX = size(X, 2)
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    xNoise = @trace(sampleNoiseFromPriorX(hyperparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorUX(hyperparams, nU, nX))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    xScale = @trace(scaleFromPriorX(hyperparams, nX))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateRealTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end

"""No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoURealT(hyperparams::HyperParameters, nU::Nothing, X::Covariates, T::Treatment)::Outcome
    n, nX = size(X)
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    T = @trace(generateRealTfromX(nothing, X, nothing, xtLS, tScale, tNoise))
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), Continuous GPSLC"""
@gen function GPSLCNoCovRealT(hyperparams::HyperParameters, nU::Int64, X::Nothing, T::Treatment)::Outcome
    n = size(hyperparams["SigmaU"], 1)
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))
    T = @trace(generateRealTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoUNoCovRealT(hyperparams::HyperParameters, nU::Nothing, X::Nothing, T::Treatment)::Outcome
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end

# Binary Valued Treatment

"""Binary Treatment GPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function GPSLCBinaryT(hyperparams::HyperParameters, nU::Int64, X::Covariates, T::Treatment)::Outcome
    n, nX = size(X)
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    xScale = @trace(scaleFromPriorX(hyperparams, nX))
    xNoise = @trace(sampleNoiseFromPriorX(hyperparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorUX(hyperparams, nU, nX))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateBinaryTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), Binary Treatment GPSLC"""
@gen function GPSLCNoCovBinaryT(hyperparams::HyperParameters, nU::Int64, X::Nothing, T::Treatment)::Outcome
    n = size(hyperparams["SigmaU"])[1]
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))
    T = @trace(generateBinaryTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))
    return Y
end

"""No latent confounders (no U), Binary Treatment GPSLC"""
@gen function GPSLCNoUBinaryT(hyperparams::HyperParameters, nU::Nothing, X::Covariates, T::Treatment)::Outcome
    n, nX = size(X)
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    T = @trace(generateBinaryTfromX(nothing, X, nothing, xtLS, tScale, tNoise))
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))
    return Y
end

"""No covariates (no X), no latent confounders (no U) for Binary Treatment GPSLC"""
@gen function GPSLCNoUNoCovBinaryT(hyperparams::HyperParameters, nU::Nothing, X::Nothing, T::Treatment)::Outcome
    n = size(T, 1)
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end
