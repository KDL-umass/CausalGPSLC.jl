export generateU, generateX, generateT, generateY

@gen function generateY(U::Nothing, X, T::Array{Bool,1}, uyLS::Nothing, xyLS, tyLS, yScale, yNoise)
    xyCovLog = sum(broadcast(rbfKernelLog, X, X, xyLS))
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(xyCovLog + tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end

@gen function generateY(U::Nothing, X::Nothing, T::Array{Bool,1}, uyLS::Nothing, xyLS::Nothing, tyLS, yScale, yNoise)
    n = size(T)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    Ycov = processCov(tyCovLog, yScale, yNoise)
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
