import FunctionalCollections

export GPSLCContinuous,
    GPSLCNoCovContinuous, GPSLCNoUContinuous, GPSLCNoCovNoUContinuous,
    GPSLCBinary,
    GPSLCNoCovBinary, GPSLCNoUBinary, GPSLCNoCovNoUBinary

"""Continous GPSLC, with Latent Confounders (U) and Covariates (X)"""
@gen function GPSLCContinuous(hyperparams, nX, nU)
    n = size(hyperparams["SigmaU"])[1]

    # uNoise, xNoise, tNoise, yNoise = @trace(sampleNoiseFromPriorUXTY(hyperparams, nothing, nX))
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    xNoise = @trace(sampleNoiseFromPriorX(hyperparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    # utLS, uyLS, uxLS, xtLS, xyLS, tyLS = @trace(lengthscaleFromPriorUTX(hyperparams, nU, nX))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # xScale = @trace(MappedGenerateScale(fill(hyperparams["xScaleShape"], nX),
    #         fill(hyperparams["xScaleScale"], nX)), :xScale)
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    xScale = @trace(scaleFromPriorX(hyperparams, nX))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, nU, n))

    # Xcov = broadcast(processCov, sum(broadcast(rbfKernelLog, U, U, uxLS)), xScale, xNoise)
    # X = @trace(MappedGenerateX(Xcov, fill(n, nX)), :X)
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))


    # utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    # xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    # Tcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    # T = @trace(mvnormal(fill(0, n), Tcov), :T)
    T = @trace(generateTfromUX(U, X, utLS, xtLS, tScale, tNoise))

    # uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    # xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))

    return Y
end

"""No Covariates (no X), Continuous GPSLC"""
@gen function GPSLCNoCovContinuous(hyperparams, nU)
    n = size(hyperparams["SigmaU"], 1)

    # uNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    # utLS, uyLS, tyLS = @trace(lengthscaleFromPriorNoX(hyperparams, nU))
    utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    # uCov = hyperparams["SigmaU"] * uNoise
    # U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))


    # utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    # Tcov = processCov(utCovLog, tScale, tNoise)
    # T = @trace(mvnormal(fill(0, n), Tcov), :T)
    T = @trace(generateTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))

    # uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(uyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))

    return Y
end


"""No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoUContinuous(hyperparams, X)
    n = length(X[1])
    nX = length(X)

    # _, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    # tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    # xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX),
    #         fill(hyperparams["xtLSScale"], nX)), :xtLS)
    # xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX),
    #         fill(hyperparams["xyLSScale"], nX)), :xyLS)
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    # xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    # Tcov = processCov(xtCovLog, tScale, tNoise)
    # T = @trace(mvnormal(fill(0, n), Tcov), :T)
    T = @trace(generateTfromX(nothing, X, nothing, xtLS, tScale, tNoise))


    # xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))

    return Y
end


"""No Covariates (no X), No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoCovNoUContinuous(hyperparams, T)
    n = length(T)

    # _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS = lengthscaleFromPriorT(hyperparams)

    #   Prior over Kernel Scale
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data     
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))

    return Y
end


"""Binary Treatment GPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function GPSLCBinary(hyperparams, nX, nU)
    n = size(hyperparams["SigmaU"])[1]

    #   Prior over Noise
    # uNoise, xNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams, nX))
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    xNoise = @trace(sampleNoiseFromPriorX(hyperparams, nX))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales    
    # utLS, uyLS, uxLS, tyLS, xtLS, xyLS = @trace(lengthscaleFromPrior(hyperparams, nU, nX))
    uxLS, utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # xScale = @trace(MappedGenerateScale(fill(hyperparams["xScaleShape"], nX),
    #         fill(hyperparams["xScaleScale"], nX)), :xScale)
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    xScale = @trace(scaleFromPriorX(hyperparams, nX))
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    # uCov = hyperparams["SigmaU"] * uNoise
    # U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))

    # Xcov = broadcast(processCov, sum(broadcast(rbfKernelLog, U, U, uxLS)), xScale, xNoise)
    # # X = @trace(MappedGenerateX(Xcov, fill(n, nX)), :X)
    X = @trace(generateXfromU(U, uxLS, xScale, xNoise, n, nX))

    # utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    # xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    # logitTcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    # logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    # T = @trace(MappedGenerateBinaryT(logitT), :T)
    T = @trace(generateTfromUX(U, X, utLS, xtLS, tScale, tNoise))

    # uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    # xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromUXT(U, X, T, uyLS, xyLS, tyLS, yScale, yNoise))

    return Y
end


"""No Covariates (no X), Binary Treatment GPSLC"""
@gen function GPSLCNoCovBinary(hyperparams, nU)
    n = size(hyperparams["SigmaU"])[1]

    #   Prior over Noise
    # uNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    uNoise = @trace(sampleNoiseFromPriorU(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))


    #   Prior over Kernel Lengthscales
    # utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
    #         fill(hyperparams["utLSScale"], nU)), :utLS)
    # uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
    #         fill(hyperparams["uyLSScale"], nU)), :uyLS)
    # tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    utLS, uyLS = @trace(lengthscaleFromPriorU(hyperparams, nU))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    # uCov = hyperparams["SigmaU"] * uNoise
    # U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    U = @trace(generateUfromSigmaU(hyperparams["SigmaU"], uNoise, n, nU))

    # utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    # logitTcov = processCov(utCovLog, tScale, tNoise)
    # logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    # T = @trace(MappedGenerateBinaryT(logitT), :T)
    T = @trace(generateTfromU(U, nothing, utLS, nothing, tScale, tNoise, n))

    # uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(uyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromUT(U, nothing, T, uyLS, nothing, tyLS, yScale, yNoise))

    return Y
end

"""No latent confounders (no U), Binary Treatment GPSLC"""
@gen function GPSLCNoUBinary(hyperparams, X)
    n = size(X[1])
    nX = size(X)

    #   Prior over Noise
    # _, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    tNoise = @trace(sampleNoiseFromPriorT(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    # tyLS, xtLS, xyLS = @trace(lengthscaleFromPriorNoU(hyperparams, nX))
    xtLS, xyLS = @trace(lengthscaleFromPriorX(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    tScale = @trace(scaleFromPriorT(hyperparams))
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    # xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    # logitTcov = processCov(xtCovLog, tScale, tNoise)
    # logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    # T = @trace(MappedGenerateBinaryT(logitT), :T)
    T = @trace(generateTfromX(nothing, X, nothing, xtLS, tScale, tNoise))

    # xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateYfromXT(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))

    return Y
end

"""No covariates (no X), no latent confounders (no U) for Binary Treatment GPSLC"""
@gen function GPSLCNoCovNoUBinary(hyperparams::HyperParameters, T::Array{Bool,1})
    n = size(T)

    #   Prior over Noise
    # _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))
    yNoise = @trace(sampleNoiseFromPriorY(hyperparams))

    #   Prior over Kernel Lengthscales
    # tyLS, xtLS, xyLS = @trace(lengthscaleFromPriorNoU(hyperparams, nX))
    tyLS = @trace(lengthscaleFromPriorT(hyperparams))

    #   Prior over Kernel Scale
    # yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    yScale = @trace(scaleFromPriorY(hyperparams))

    #   Generate Data 
    Y = @trace(generateYfromT(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end
