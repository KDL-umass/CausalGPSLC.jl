module Model

# +
using Gen
using LinearAlgebra
import Base.show

export x_kernel, y_kernel, GPROC, AdditiveNoiseGPROC

# +
function x_kernel(u1::Float64, u2::Float64, uxLS::Float64, eps1::Float64, eps2::Float64, 
                  epsxLS::Float64, xScale::Float64)
    u_term = ((u1 - u2)/uxLS)^2
    eps_term = ((eps1 - eps2)/epsxLS)^2
    return xScale * exp(-(u_term + eps_term)/2)
end

function x_kernel(u1::Float64, u2::Float64, uxLS::Float64, xScale::Float64)
    u_term = ((u1 - u2)/uxLS)^2
    return xScale * exp(-(u_term)/2)
end
    
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, x1::Float64, x2::Float64,
                  xyLS::Float64, eps1::Float64, eps2::Float64, epsyLS::Float64, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    eps_term = ((eps1 - eps2)/epsyLS)^2
    return yScale * exp(-(u_term + x_term + eps_term)/2)
end

function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, x1::Float64, x2::Float64, 
                    xyLS::Float64, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    return yScale * exp(-(u_term + x_term)/2)
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
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(uniform(hyperparams["uNoiseMin"], hyperparams["uNoiseMin"]), :uNoise)
    xNoise = @trace(uniform(hyperparams["xNoiseMin"], hyperparams["xNoiseMax"]), :xNoise)
    yNoise = @trace(uniform(hyperparams["yNoiseMin"], hyperparams["yNoiseMax"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    uxLS = @trace(inv_gamma(hyperparams["uxLSShape"], hyperparams["uxLSScale"]), :uxLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS)
    xyLS = @trace(inv_gamma(hyperparams["xyLSShape"], hyperparams["xyLSScale"]), :xyLS)    
    
#   Prior over Kernel Scale
    xScale = @trace(inv_gamma(hyperparams["xScaleShape"], hyperparams["xScaleScale"]), :xScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    Xcov = broadcast(x_kernel, U, U', uxLS, xScale) + xNoise * 1I
    X = @trace(mvnormal(fill(0, n), Xcov), :X)
    
    Ycov = broadcast(y_kernel, U, U', uyLS, X, X', xyLS, yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
# -
end
