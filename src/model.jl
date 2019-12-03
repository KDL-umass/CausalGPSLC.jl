module Model

# +
using Gen
using LinearAlgebra
import Base.show

export x_kernel, y_kernel, GPROC, AdditiveNoiseGPROC

# +
function x_kernel(u1::Float64, u2::Float64, uxLS::Float64, eps1::Float64, eps2::Float64, epsxLS::Float64)
    u_term = ((u1 - u2)/uxLS)^2
    eps_term = ((eps1 - eps2)/epsxLS)^2
    return exp(-(u_term + eps_term)/2)
end

function x_kernel(u1::Float64, u2::Float64, uxLS::Float64)
# function x_kernel(u1, u2, uxLS)
    u_term = ((u1 - u2)/uxLS)^2
    return exp(-(u_term)/2)
end
    
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, x1::Float64, x2::Float64,
                  xyLS::Float64, eps1::Float64, eps2::Float64, epsyLS::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    eps_term = ((eps1 - eps2)/epsyLS)^2
    return exp(-(u_term + x_term + eps_term)/2)
end

function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, x1::Float64, x2::Float64, xyLS::Float64)
# function y_kernel(u1, u2, uyLS, x1, x2, xyLS)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    return exp(-(u_term + x_term)/2)
end

# +
@gen (grad, static) function generateEps(noise::Float64)
    eps = @trace(normal(0, noise), :eps)
    return eps
end

MappedGenerateEps = Map(generateEps)

@gen (static) function GPROC(hyperparams)    
    n = size(hyperparams["uCov"])[1]
    
    U = @trace(mvnormal(fill(0, n), hyperparams["uCov"]), :U)
    
    epsX = @trace(MappedGenerateEps(fill(hyperparams["xNoise"], n)), :epsX)
    Xcov = broadcast(x_kernel, U, U', hyperparams["uxLS"], epsX, epsX', hyperparams["epsxLS"])
    X = @trace(mvnormal(fill(0, n), Xcov), :X)
    
    epsY = @trace(MappedGenerateEps(fill(hyperparams["yNoise"], n)), :epsY)
    Ycov = broadcast(y_kernel, U, U', hyperparams["uyLS"], 
                               X, X', hyperparams["xyLS"], 
                               epsY, epsY', hyperparams["epsyLS"])
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end

@gen (static) function AdditiveNoiseGPROC(hyperparams)    
    n = size(hyperparams["uCov"])[1]
    
    U = @trace(mvnormal(fill(0, n), hyperparams["uCov"]), :U)
    
    Xcov = broadcast(x_kernel, U, U', hyperparams["uxLS"]) + hyperparams["xNoise"] * 1I
    X = @trace(mvnormal(fill(0, n), Xcov), :X)
    
    Ycov = broadcast(y_kernel, U, U', hyperparams["uyLS"], X, X', hyperparams["xyLS"]) + hyperparams["yNoise"] * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
# -
end
