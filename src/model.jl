import FunctionalCollections

export GPSLCRealT,
    GPSLCNoURealT, GPSLCNoCovRealT, GPSLCRealT,
    GPSLCBinaryT,
    GPSLCNoUBinaryT, GPSLCNoCovBinaryT, GPSLCNoUNoCovBinaryT

"""Continous GPSLC, with Latent Confounders (U) and Covariates (X)"""
@gen function GPSLCRealT(hyperparams::HyperParameters, nU::Int64, nX::Int64)
    n = size(hyperparams["SigmaU"])[1]
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
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, nU, n))
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateRealTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end

"""No Covariates (no X), Continuous GPSLC"""
@gen function GPSLCNoCovRealT(hyperparams::HyperParameters, nU::Int64)
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


"""No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoURealT(hyperparams::HyperParameters, X::Covariates)
    n = length(X[1])
    nX = length(X)
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


"""No Covariates (no X), No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoUNoCovRealT(hyperparams::HyperParameters, T::Treatment)
    n = length(T)
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    tyLS = lengthscaleFromPriorT(hyperparams)
    yScale = @trace(scaleFromPriorY(hyperparams))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))

    return Y
end


"""Binary Treatment GPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function GPSLCBinaryT(hyperparams::HyperParameters, nU::Int64, nX::Int64)
    n = size(hyperparams["SigmaU"], 1)
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
    println("uxLS $(size(uxLS))")
    println("U $(size(U))")
    println("n $n")
    println("nX $nX")
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))
    T = @trace(generateBinaryTfromUX(U, X, utLS, xtLS, tScale, tNoise))
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))
    return Y
end


"""No Covariates (no X), Binary Treatment GPSLC"""
@gen function GPSLCNoCovBinaryT(hyperparams::HyperParameters, nU::Int64)
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
@gen function GPSLCNoUBinaryT(hyperparams::HyperParameters, X::Covariates)
    nX = size(X, 2)
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
@gen function GPSLCNoUNoCovBinaryT(hyperparams::HyperParameters, T::Treatment)
    n = size(T)
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end
