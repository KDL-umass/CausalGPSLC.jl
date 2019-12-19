module Model

# +
using Gen
using LinearAlgebra
import Base.show
import FunctionalCollections

export rbfKernel, linearKernel, ContinuousGPROC, BinaryGPROC

# +
function rbfKernel(x1::Float64, x2::Float64, LS::Float64)
    return exp(-((x1 - x2)/LS)^2)
end

function rbfKernel(x1::Bool, x2::Bool, LS::Float64)
    return exp(-((x1 - x2)/LS)^2)
end

function rbfKernel(xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, LS)
        return exp(-(sum((xpair[1] - xpair[2])./LS).^2))
end

function rbfKernel(xpair::Nothing, LS)
    return 1
end

function linearKernel(x1::Float64, x2::Float64, LS::Float64)
    return x1 * x2/LS
end

function linearKernel(xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, LS)
        return sum((xpair[1] - xpair[2])./LS)
end

function linearKernel(xpair::Nothing, LS)
        return 1
end

expit(x::Real) = exp(x) / (1.0 + exp(x))

# +
@gen (static) function generateLS(shape, scale)
    LS = @trace(inv_gamma(shape, scale), :LS)
    return LS
end

@gen (static) function generateBinaryT(logitT)
    Tr = @trace(bernoulli(expit(logitT)), :Tr)
    return Tr
end
    
MappedGenerateLS = Map(generateLS)
MappedGenerateBinaryT = Map(generateBinaryT)

load_generated_functions()

@gen (static) function ContinuousGPROC(hyperparams, Xcol, nX)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(inv_gamma(hyperparams["utLSShape"], hyperparams["utLSScale"]), :utLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX), 
                                      fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX), 
                                      fill(hyperparams["xyLSScale"], nX)), :xyLS)
    
    
#   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    Tcov = (broadcast(hyperparams["utKernel"], U, U', utLS) .* 
            broadcast(hyperparams["xtKernel"], Xcol, (xtLS,)) * tScale) + tNoise * 1I
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    Ycov = (broadcast(hyperparams["uyKernel"], U, U', uyLS) .* 
            broadcast(hyperparams["xyKernel"], Xcol, (xtLS,)) .* 
            broadcast(hyperparams["tyKernel"], Tr, Tr', tyLS) * yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end

@gen (static) function BinaryGPROC(hyperparams, Xcol, nX)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(inv_gamma(hyperparams["utLSShape"], hyperparams["utLSScale"]), :utLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX), 
                                      fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX), 
                                      fill(hyperparams["xyLSScale"], nX)), :xyLS)
    
#   Prior over Kernel Scale
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    logitTCov = (broadcast(hyperparams["utKernel"], U, U', utLS) .* 
            broadcast(hyperparams["xtKernel"], Xcol, (xtLS,)) * tScale) + tNoise * 1I
    logitT = @trace(mvnormal(fill(0, n), logitTCov), :logitT)
    Tr = @trace(MappedGenerateBinaryT(logitT), :Tr)
    
    Ycov = (broadcast(hyperparams["uyKernel"], U, U', uyLS) .* 
            broadcast(hyperparams["xyKernel"], Xcol, (xtLS,)) .* 
            broadcast(hyperparams["tyKernel"], Tr, Tr', tyLS) * yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
# -
end
