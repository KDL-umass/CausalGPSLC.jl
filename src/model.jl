import FunctionalCollections

export GPSLCContinuous,
    GPSLCNoCovContinuous, GPSLCNoUContinuous, GPSLCNoCovNoUContinuous,
    GPSLCBinary,
    GPSLCNoCovBinary, GPSLCNoUBinary, GPSLCNoCovNoUBinary

"""Gen function to generate lengthscale parameter for GP"""
@gen function generateLS(shape, scale)
    @trace(inv_gamma(shape, scale), :LS)
end

"""Gen function to generate scale parameter for GP"""
@gen function generateScale(shape, scale)
    @trace(inv_gamma(shape, scale), :Scale)
end

"""Gen function to generate noise from inv_gamma"""
@gen function generateNoise(shape, scale)
    @trace(inv_gamma(shape, scale), :Noise)
end

"""Gen function to generate binary treatment (T)"""
@gen function generateBinaryT(logitT)
    @trace(bernoulli(expit(logitT)), :T)
end

"""Gen function to generate latent confounders (U) from mvnormal distribution"""
@gen function generateU(Ucov::Array{Float64}, n::Int)
    @trace(mvnormal(fill(0, n), Ucov), :U)
end

"""Gen function to generate covariates (X) from mvnormal distribution"""
@gen function generateX(Xcov::Array{Float64}, n::Int)
    @trace(mvnormal(fill(0, n), Xcov), :X)
end

MappedGenerateLS = Map(generateLS)
MappedMappedGenerateLS = Map(MappedGenerateLS)
MappedGenerateScale = Map(generateScale)
MappedGenerateBinaryT = Map(generateBinaryT)
MappedGenerateNoise = Map(generateNoise)
MappedGenerateU = Map(generateU)
MappedGenerateX = Map(generateX)

load_generated_functions()

"""
Generate noise terms from noise prior
"""
@gen function sampleNoiseFromPrior(hyperparams, nX)
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    xNoise = @trace(MappedGenerateNoise(fill(hyperparams["xNoiseShape"], nX),
            fill(hyperparams["xNoiseScale"], nX)), :xNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)

    return uNoise, xNoise, tNoise, yNoise
end

@gen function sampleNoiseFromPrior(hyperparams)
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    return uNoise, tNoise, yNoise
end

"""Treatment to outcome lengthscale"""
@gen function sampleTYLengthscale(hyperparams)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    return tyLS
end

"""Latent confounders to treatment and outcome lengthscale"""
@gen function sampleUtUyLengthscale(hyperparams)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    return utLS, uyLS
end

"""Covariates to treatment and outcome lengthscale"""
@gen function sampleXtXyLengthscale(hyperparams, nX)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX),
            fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX),
            fill(hyperparams["xyLSScale"], nX)), :xyLS)
    return xtLS, xyLS
end


"""
Generate kernel lengthscales from prior
"""
@gen function lengthscaleFromPrior(hyperparams, nU, nX)
    utLS, uyLS = @trace(sampleUtUyLengthscale(hyperparams))

    uxLS = @trace(MappedMappedGenerateLS(fill(fill(hyperparams["uxLSShape"], nX), nU),
            fill(fill(hyperparams["uxLSScale"], nX), nU)), :uxLS)

    tyLS = @trace(sampleTYLengthscale(hyperparams))
    xtLS, xyLS = sampleXtXyLengthscale(hyperparams, nX)

    return utLS, uyLS, uxLS, tyLS, xtLS, xyLS
end

"""
Generate kernel lengthscales from prior without covariates
"""
@gen function lengthscaleFromPriorNoX(hyperparams, nU)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    return utLS, uyLS, tyLS
end

"""
Generate kernel lengthscales from prior without latent confounders
"""
@gen function lengthscaleFromPriorNoU(hyperparams, nX)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    xtLS, xyLS = @trace(sampleXtXyLengthscale(hyperparams, nX))
    return tyLS, xtLS, xyLS
end

"""
Generate kernel lengthscales from prior without latent confounders or covariates
"""
@gen function lengthscaleFromPriorNoUNoX(hyperparams)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    return tyLS
end

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
    uCov = hyperparams["SigmaU"] * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    Xcov = broadcast(processCov, sum(broadcast(rbfKernelLog, U, U, uxLS)), xScale, xNoise)

    X = @trace(MappedGenerateX(Xcov, fill(n, nX)), :X)
    utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    Tcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(fill(0, n), Tcov), :T)
    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

    return Y
end

"""No Covariates (no X), Continuous GPSLC"""
@gen function GPSLCNoCovContinuous(hyperparams, nU)
    n = size(hyperparams["SigmaU"])[1]

    uNoise, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    utLS, uyLS, tyLS = @trace(lengthscaleFromPriorNoX(hyperparams, nU))

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    uCov = hyperparams["SigmaU"] * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)

    utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    Tcov = processCov(utCovLog, tScale, tNoise)
    T = @trace(mvnormal(fill(0, n), Tcov), :T)

    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

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
    xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    Tcov = processCov(xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(fill(0, n), Tcov), :T)

    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

    return Y
end


"""No Covariates (no X), No Latent Confounders (no U), Continuous GPSLC"""
@gen function GPSLCNoCovNoUContinuous(hyperparams, T)
    n = length(T)

    _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS = lengthscaleFromPriorNoUNoX(hyperparams)

    #   Prior over Kernel Scale
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data     
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

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
    uCov = hyperparams["SigmaU"] * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    Xcov = broadcast(processCov, sum(broadcast(rbfKernelLog, U, U, uxLS)), xScale, xNoise)

    X = @trace(MappedGenerateX(Xcov, fill(n, nX)), :X)
    utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    logitTcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

    return Y
end


"""No Covariates (no X), Binary Treatment GPSLC"""
@gen function GPSLCNoCovBinary(hyperparams, nU)
    n = size(hyperparams["SigmaU"])[1]

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
    uCov = hyperparams["SigmaU"] * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)

    utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
    logitTcov = processCov(utCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    uyCovLog = sum(broadcast(rbfKernelLog, U, U, uyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)

    return Y
end

"""No latent confounders (no U), Binary Treatment GPSLC"""
@gen function GPSLCNoUBinary(hyperparams, X)
    n = size(X[1])
    nX = size(X)

    #   Prior over Noise
    _, tNoise, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS, xtLS, xyLS = @trace(lengthscaleFromPriorNoU(hyperparams, nX))

    #   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    logitTcov = processCov(xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)

    # xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    # tyCovLog = rbfKernelLog(T, T, tyLS)
    # Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    # Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    Y = @trace(generateY(nothing, X, T, nothing, xyLS, tyLS, yScale, yNoise))

    return Y
end

"""No covariates (no X), no latent confounders (no U) for Binary Treatment GPSLC"""
@gen function GPSLCNoCovNoUBinary(hyperparams::HyperParameters, T::Array{Bool,1})
    n = size(T)

    #   Prior over Noise
    _, _, yNoise = @trace(sampleNoiseFromPrior(hyperparams))

    #   Prior over Kernel Lengthscales
    tyLS, xtLS, xyLS = @trace(lengthscaleFromPriorNoU(hyperparams, nX))

    #   Prior over Kernel Scale
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)

    #   Generate Data 
    Y = @trace(generateY(nothing, nothing, T, nothing, nothing, tyLS, yScale, yNoise))
    return Y
end
