export GPSLCContinuous,
    GPSLCNoCovContinuous, GPSLCNoUContinuous, GPSLCNoCovNoUContinuous,
    GPSLCBinary,
    GPSLCNoCovBinary, GPSLCNoUBinary, GPSLCNoCovNoUBinary

"""Continous GPSLC, with Latent Confounders (U) and Covariates (X)"""
@gen function GPSLCContinuous(hyperparams, nX, nU)
    n = size(hyperparams["SigmaU"])[1]

    uNoise, xNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams, nX))

    #   Prior over Kernel Lengthscales
    utLS, uyLS, uxLS, tyLS, xtLS, xyLS = @trace(lengthscaleFromPrior(hyperparams, nU, nX))

    #   Prior over Kernel Scale
    xScale = @trace(MappedGenerateScale(fill(hyperparams["xScaleShape"], nX),
            fill(hyperparams["xScaleScale"], nX)), :xScale)
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, nothing, hyperparams["SigmaU"], uNoise, nU, uxLS, xScale, xNoise, nX, utLS, xtLS, tScale, tNoise, uyLS, xyLS, tyLS, yScale, yNoise))

    return Y
end


"""No Covariates (no X), Continuous GPSLC"""
@gen function GPSLCNoCovContinuous(hyperparams::HyperParameters, nU::Int64)
    uNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    utLS, uyLS, tyLS = @trace(lengthscaleFromPrior(hyperparams, nU, nothing))

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, nothing, hyperparams["SigmaU"], uNoise, nU, nothing, nothing, nothing, nothing, utLS, nothing, tScale, tNoise, uyLS, nothing, tyLS, yScale, yNoise))

    return Y
end


"""No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoUContinuous(hyperparams, X)
    n = length(X[1])
    nX = length(X)

    _, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX),
            fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX),
            fill(hyperparams["xyLSScale"], nX)), :xyLS)

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(X, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, xtLS, tScale, tNoise, nothing, xyLS, tyLS, yScale, yNoise))

    return Y
end



"""No Covariates (no X), No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoCovNoUContinuous(hyperparams, T)
    n = length(T)

    _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS = lengthscaleFromPrior(hyperparams, nothing, nothing)

    #   Prior over Kernel Scale
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data     
    Y = @trace(generateY(nothing, T, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, tyLS, yScale, yNoise))

    return Y
end


"""Binary Treatment GPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function GPSLCBinary(hyperparams, nX, nU)
    n = size(hyperparams["SigmaU"])[1]

    #   Prior over Noise
    uNoise, xNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams, nX))

    #   Prior over Kernel Lengthscales    
    utLS, uyLS, uxLS, tyLS, xtLS, xyLS = @trace(lengthscaleFromPrior(hyperparams, nU, nX))

    #   Prior over Kernel Scale
    xScale = @trace(MappedGenerateScale(fill(hyperparams["xScaleShape"], nX),
            fill(hyperparams["xScaleScale"], nX)), :xScale)
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, nothing, hyperparams["SigmaU"], uNoise, nU, uxLS, xScale, xNoise, nX, utLS, xtLS, tScale, tNoise, uyLS, xyLS, tyLS, yScale, yNoise))

    return Y
end


"""No Covariates (no X), Binary Treatment GPSLC"""
@gen function GPSLCNoCovBinary(hyperparams, nU)
    n = size(hyperparams["SigmaU"], 1)

    #   Prior over Noise
    uNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))


    #   Prior over Kernel Lengthscales
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, T, hyperparams["SigmaU"], nU, uNoise, nothing, nothing, nothing, nothing, utLS, nothing, tScale, tNoise, uyLS, nothing, tyLS, yScale, yNoise))

    return Y
end


"""No latent confounders (no U), Binary Treatment GPSLC"""
@gen function GPSLCNoUBinary(hyperparams, X)
    n = size(X[1])
    nX = size(X)

    #   Prior over Noise
    _, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS, xtLS, xyLS = @trace(lengthscaleFromPriorNoU(hyperparams, nothing, nX))

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(X, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, xtLS, tScale, tNoise, nothing, xyLS, tyLS, yScale, yNoise))

    return Y
end


"""No covariates (no X), no latent confounders (no U) for Binary Treatment GPSLC"""
@gen function GPSLCNoCovNoUBinary(hyperparams, T::Array{Bool,1})
    n = size(T)

    #   Prior over Noise
    _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS, _, _ = @trace(lengthscaleFromPriorNoU(hyperparams, nothing, nX))

    #   Prior over Kernel Scale
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, T, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, tyLS, yScale, yNoise))

    return Y
end
