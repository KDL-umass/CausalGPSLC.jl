module Model

# +
using Gen
using LinearAlgebra
import Base.show
import FunctionalCollections

export processCov, rbfKernelLog, ContinuousGPROC, BinaryGPROC

# +
function rbfKernelLog(X1::Array{Float64, 1}, X2::Array{Float64, 1}, LS::Array{Float64, 1})
    return -broadcast(/, ((X1 .- X2').^2,), LS.^2)
end

function rbfKernelLog(X1::Array{Float64, 1}, X2::Array{Float64, 1}, 
                      LS::FunctionalCollections.PersistentVector{Float64})
    return -broadcast(/, ((X1 .- X2').^2,), LS.^2)
end

function rbfKernelLog(X1::Array{Float64, 1}, X2::Array{Float64, 1}, LS::Float64)
    return -((X1 .- X2')/LS).^2
end

function rbfKernelLog(X1::Array{Bool, 1}, X2::Array{Bool, 1}, LS::Float64)
    return -((X1 .- X2')/LS).^2
end


function rbfKernelLog(X1::FunctionalCollections.PersistentVector{Bool}, 
                      X2::FunctionalCollections.PersistentVector{Bool}, LS::Float64)
    return -((X1 .- X2')/LS).^2
end


expit(x::Real) = exp(x) / (1.0 + exp(x))

function processCov(logCov::Array{Float64}, scale::Float64, noise::Float64)
    return exp.(logCov) * scale + 1I*noise
end

# +
@gen (static) function generateLS(shape, scale)
    LS = @trace(inv_gamma(shape, scale), :LS)
    return LS
end

@gen (static) function generateScale(shape, scale)
    Scale = @trace(inv_gamma(shape, scale), :Scale)
    return Scale
end

@gen (static) function generateNoise(shape, scale)
    Noise = @trace(inv_gamma(shape, scale), :Noise)
    return Noise
end

@gen (static) function generateBinaryT(logitT)
    T = @trace(bernoulli(expit(logitT)), :T)
    return T
end

@gen (static) function generateU(Ucov::Array{Float64}, n::Int)
    U = @trace(mvnormal(fill(0, n), Ucov), :U)
    return U
end

@gen (static) function generateX(Xcov::Array{Float64}, n::Int)
    X = @trace(mvnormal(fill(0, n), Xcov), :X)
    return X
end
    
MappedGenerateLS = Map(generateLS)
MappedMappedGenerateLS = Map(MappedGenerateLS)
MappedGenerateScale = Map(generateScale)
MappedGenerateBinaryT = Map(generateBinaryT)
MappedGenerateNoise = Map(generateNoise)
MappedGenerateU = Map(generateU)
MappedGenerateX = Map(generateX)

load_generated_functions()

@gen (static) function ContinuousGPROC(hyperparams, nX, nU)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    xNoise = @trace(MappedGenerateNoise(fill(hyperparams["xNoiseShape"], nX),
                                        fill(hyperparams["xNoiseScale"], nX)), :xNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU), 
                                   fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU), 
                                   fill(hyperparams["uyLSScale"], nU)), :uyLS)
    uxLS = @trace(MappedMappedGenerateLS(fill(fill(hyperparams["uxLSShape"], nX), nU), 
                                         fill(fill(hyperparams["uxLSScale"], nX), nU)), :uxLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX), 
                                   fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX), 
                                   fill(hyperparams["xyLSScale"], nX)), :xyLS)
    
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

    return U
end
# -
@gen (static) function BinaryGPROC(hyperparams, nX, nU)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    xNoise = @trace(MappedGenerateNoise(fill(hyperparams["xNoiseShape"], nX),
                                        fill(hyperparams["xNoiseScale"], nX)), :xNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU), 
                                   fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU), 
                                   fill(hyperparams["uyLSScale"], nU)), :uyLS)
    uxLS = @trace(MappedMappedGenerateLS(fill(fill(hyperparams["uxLSShape"], nX), nU), 
                                         fill(fill(hyperparams["uxLSScale"], nX), nU)), :uxLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX), 
                                   fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX), 
                                   fill(hyperparams["xyLSScale"], nX)), :xyLS)
    
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

end


