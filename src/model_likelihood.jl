export generateUfromSigmaU, generateXfromU, generateBinaryTfromUX, generateBinaryTfromU, generateRealTfromU, generateBinaryTfromX, generateYfromUXT, generateYfromUT, generateYfromXT, generateYfromT

"""Sample U"""
@gen function generateUfromSigmaU(SigmaU, uNoise, n::Int64, nU::Int64)
    uCov = SigmaU * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    U = toMatrix(U, n, nU)
    @assert size(U) == (n, nU)
    return U
end

"""Sample X from U"""
@gen function generateXfromU(U::Confounders, uxLS::SupportedRBFLengthscale, xScale::XScaleOrNoise, xNoise::XScaleOrNoise, n::Int64, nX::Int64)
    X = zeros(n, nX)
    for k = 1:nX
        uxCovLog_k = rbfKernelLog(U, U, uxLS[k, :])
        xCov_k = processCov(uxCovLog_k, xScale[k], xNoise[k])
        @assert size(xCov_k) == (n, n) "X covariance matrix for dim k"
        X[:, k] = @trace(generateX(xCov_k, n), :X => k)
    end
    return X
end

"""Sample Binary T from confounders (U) and covariates (X)"""
@gen function generateBinaryTfromUX(U::Any, X::Any, utLS::SupportedRBFLengthscale, xtLS::SupportedRBFLengthscale, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    utCovLog = rbfKernelLog(U, U, utLS)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    logitTCov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTCov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample T from confounders (U) and covariates (X)"""
@gen function generateRealTfromUX(U::Any, X::Any, utLS::SupportedRBFLengthscale, xtLS::SupportedRBFLengthscale, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    utCovLog = rbfKernelLog(U, U, utLS)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    tCov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), tCov), :T)
    return T
end

"""Sample Binary T from confounders (U)"""
@gen function generateBinaryTfromU(U::Any, X::Nothing, utLS::SupportedRBFLengthscale, xtLS::Nothing, tScale::Float64, tNoise::Float64, n::Int64)
    utCovLog = rbfKernelLog(U, U, utLS)
    logitTCov = processCov(utCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTCov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample Continuous T from confounders (U)"""
@gen function generateRealTfromU(U::Any, X::Nothing, utLS::SupportedRBFLengthscale, xtLS::Nothing, tScale::Float64, tNoise::Float64, n::Int64)
    utCovLog = rbfKernelLog(U, U, utLS)
    tCov = processCov(utCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), tCov), :T)
    return T
end

"""Sample Binary T from covariates (X)"""
@gen function generateBinaryTfromX(U::Nothing, X::Covariates, utLS::Nothing, xtLS::SupportedRBFLengthscale, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    @assert size(xtCovLog) == (n, n) "tCov needs to be NxN!"
    logitTCov = processCov(xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTCov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample Binary T from covariates (X)"""
@gen function generateRealTfromX(U::Nothing, X::Any, utLS::Nothing, xtLS::SupportedRBFLengthscale, tScale::Float64, tNoise::Float64)
    n, nX = size(X)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    tCov = processCov(xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), tCov), :T)
    return T
end

"""Sample Y from confounders (U), covariates (X), and treatment (T)"""
@gen function generateYfromUXT(U::Any, X::Any, T::Any, uyLS::SupportedRBFLengthscale, xyLS::SupportedRBFLengthscale, tyLS::SupportedRBFLengthscale, yScale::Float64, yNoise::Float64)
    n = size(T, 1)
    uyCovLog = rbfKernelLog(U, U, uyLS)
    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog .+ xyCovLog .+ tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end

"""Sample Y from confounders (U) and treatment (T)"""
@gen function generateYfromUT(U::Any, X::Nothing, T::Treatment, uyLS::SupportedRBFLengthscale, xyLS::Nothing, tyLS::SupportedRBFLengthscale, yScale::Float64, yNoise::Float64)
    n = size(T, 1)
    uyCovLog = rbfKernelLog(U, U, uyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog .+ tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end

"""Sample Y from covariates (X) and treatment (T)"""
@gen function generateYfromXT(U::Nothing, X::Covariates, T::Treatment, uyLS::Nothing, xyLS::SupportedRBFLengthscale, tyLS::SupportedRBFLengthscale, yScale::Float64, yNoise::Float64)
    n = size(T, 1)
    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(xyCovLog .+ tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end

"""Sample Y from only treatment (T)"""
@gen function generateYfromT(U::Nothing, X::Nothing, T::Treatment, uyLS::Nothing, xyLS::Nothing, tyLS::SupportedRBFLengthscale, yScale::Float64, yNoise::Float64)
    n = size(T, 1)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end
