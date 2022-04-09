export generateUfromSigmaU, generateXfromU, generateBinaryTfromUX, generateBinaryTfromU, generateRealTfromU, generateBinaryTfromX, generateYfromUXT, generateYfromUT, generateYfromXT, generateYfromT

"""Sample U"""
@gen function generateUfromSigmaU(SigmaU, uNoise, n, nU)
    uCov = SigmaU * uNoise
    U = @trace(MappedGenerateU(fill(uCov, nU), fill(n, nU)), :U)
    U = toMatrix(U, n, nU)
    @assert size(U) == (n, nU)
    return U
end

"""Sample X from U"""
@gen function generateXfromU(U::Any, uxLS, xScale, xNoise, n, nX)
    Xcov = broadcast(processCov, sum(broadcast(rbfKernelLog, U, U, uxLS)), xScale, xNoise)
    X = @trace(MappedGenerateX(Xcov, fill(n, nX)), :X)
    return X
end

"""Sample Binary T from confounders (U) and covariates (X)"""
@gen function generateBinaryTfromUX(U::Any, X::Any, utLS, xtLS, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    utCovLog = rbfKernelLog(U, U, utLS)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    logitTcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample T from confounders (U) and covariates (X)"""
@gen function generateRealTfromUX(U::Any, X::Any, utLS, xtLS, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    utCovLog = rbfKernelLog(U, U, utLS)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    Tcov = processCov(utCovLog + xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), Tcov), :T)
    return T
end

"""Sample Binary T from confounders (U)"""
@gen function generateBinaryTfromU(U::Any, X::Nothing, utLS, xtLS::Nothing, tScale::Float64, tNoise::Float64, n::Int64)
    utCovLog = rbfKernelLog(U, U, utLS)
    logitTcov = processCov(utCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample Continuous T from confounders (U)"""
@gen function generateRealTfromU(U::Any, X::Nothing, utLS, xtLS::Nothing, tScale::Float64, tNoise::Float64, n::Int64)
    utCovLog = rbfKernelLog(U, U, utLS)
    Tcov = processCov(utCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), Tcov), :T)
    return T
end

"""Sample Binary T from covariates (X)"""
@gen function generateBinaryTfromX(U::Nothing, X::Union{Matrix{Float64},Vector{Float64}}, utLS::Nothing, xtLS, tScale::Float64, tNoise::Float64)
    n = size(X, 1)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    @assert size(xtCovLog) == (n, n) "tCov needs to be NxN!"
    logitTcov = processCov(xtCovLog, tScale, tNoise)
    logitT = @trace(mvnormal(zeros(n), logitTcov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end

"""Sample Binary T from covariates (X)"""
@gen function generateRealTfromX(U::Nothing, X::Any, utLS::Nothing, xtLS, tScale::Float64, tNoise::Float64)
    xtCovLog = rbfKernelLog(X, X, xtLS)
    Tcov = processCov(xtCovLog, tScale, tNoise)
    T = @trace(mvnormal(zeros(n), Tcov), :T)
    return T
end

"""Sample Y from confounders (U), covariates (X), and treatment (T)"""
@gen function generateYfromUXT(U::Any, X::Any, T::Any, uyLS::Float64, xyLS::Float64, tyLS::Float64, yScale::Float64, yNoise::Float64)
    n = size(T, 1)
    uyCovLog = rbfKernelLog(U, U, uyLS)
    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(uyCovLog .+ xyCovLog .+ tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end

"""Sample Y from confounders (U) and treatment (T)"""
@gen function generateYfromUT(U::Any, X::Nothing, T::Treatment, uyLS, xyLS::Nothing, tyLS::Float64, yScale::Float64, yNoise::Float64)
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
    println("n $n")
    println("T $(size(T)) $(T)")
    println("tyLS $(size(tyLS)) $tyLS")
    tyCovLog = rbfKernelLog(T, T, tyLS)
    println("tyCovLog $(size(tyCovLog))")
    Ycov = processCov(tyCovLog, yScale, yNoise)
    println("Ycov $(size(Ycov))")
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)
    return Y
end
