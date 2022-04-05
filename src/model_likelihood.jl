export generateY
"""
Generate output (Y) from parameters

Continous GPSLC, with Latent Confounders (U) and Covariates (X)
"""
@gen function generateY(X::Nothing, T::Nothing, SigmaU, uNoise, nU, uxLS, xScale, xNoise, nX, utLS, xtLS, tScale, tNoise, uyLS, xyLS, tyLS, yScale, yNoise)
    n = size(SigmaU, 1)
    uCov = SigmaU * uNoise
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
@gen function generateY(X::Nothing, T::Nothing, SigmaU, uNoise, nU, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS, xtLS::Nothing, tScale, tNoise, uyLS, xyLS::Nothing, tyLS, yScale, yNoise)
    println("Generate Y 181")
    n = size(SigmaU, 1)
    uCov = SigmaU * uNoise
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
@gen function generateY(X, T::Nothing, SigmaU::Nothing, uNoise::Nothing, nU::Nothing, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS::Nothing, xtLS, tScale, tNoise, uyLS::Nothing, xyLS, tyLS, yScale, yNoise)
    n = length(X[1])
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
@gen function generateY(X::Nothing, T, SigmaU::Nothing, uNoise::Nothing, nU::Nothing, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS::Nothing, xtLS::Nothing, tScale::Nothing, tNoise::Nothing, uyLS::Nothing, xyLS::Nothing, tyLS, yScale, yNoise)
    n = length(T)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end


"""Binary Treatment GPSLC with Covariates (X) and Latent Confounders (U)"""
@gen function generateY(X::Nothing, T::Nothing, SigmaU, uNoise, nU, uxLS, xScale, xNoise, nX, utLS, xtLS, tScale, tNoise, uyLS, xyLS, tyLS, yScale, yNoise)
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
@gen function generateY(X::Nothing, T, SigmaU, uNoise, nU, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS, xtLS::Nothing, tScale, tNoise, uyLS, xyLS::Nothing, tyLS, yScale, yNoise)
    uCov = SigmaU * uNoise
    n = size(SigmaU, 1)
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
@gen function generateY(X, T::Nothing, SigmaU::Nothing, uNoise::Nothing, nU::Nothing, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS::Nothing, xtLS, tScale, tNoise, uyLS::Nothing, xyLS, tyLS, yScale, yNoise)
    xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
    logitTcov = processCov(xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(fill(0, n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end


"""No covariates (no X), no latent confounders (no U) for Binary Treatment GPSLC"""
@gen function generateY(X::Nothing, T, SigmaU::Nothing, uNoise::Nothing, nU::Nothing, uxLS::Nothing, xScale::Nothing, xNoise::Nothing, nX::Nothing, utLS::Nothing, xtLS::Nothing, tScale::Nothing, tNoise::Nothing, uyLS::Nothing, xyLS::Nothing, tyLS, yScale, yNoise)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
end
