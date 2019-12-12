module Model

# +
using Gen
using LinearAlgebra
import Base.show

export t_kernel, y_kernel, GPROC, AdditiveNoiseGPROC, LinearAdditiveNoiseGPROC

# +
function t_kernel(u1::Float64, u2::Float64, utLS::Float64, eps1::Float64, eps2::Float64, 
                  epsxLS::Float64, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    eps_term = ((eps1 - eps2)/epstLS)^2
    return tScale * exp(-(u_term + eps_term)/2)
end

function t_kernel(u1::Float64, u2::Float64, utLS::Float64, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    return tScale * exp(-(u_term)/2)
end
    
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64,
                  tyLS::Float64, eps1::Float64, eps2::Float64, epsyLS::Float64, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    eps_term = ((eps1 - eps2)/epsyLS)^2
    return yScale * exp(-(u_term + t_term + eps_term)/2)
end

function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64, 
                    tyLS::Float64, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    return yScale * exp(-(u_term + t_term)/2)
end
# This gives a linear kernel for ty relationship.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    return yScale * (exp(-u_term/2) + t1 * t2)
end

# +
@gen (grad, static) function generateEps(noise::Float64)
    eps = @trace(normal(0, noise), :eps)
    return eps
end

MappedGenerateEps = Map(generateEps)

@gen (static) function GPROC(hyperparams)    
    n = size(hyperparams["SigmaU"])[1]
    
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"]), :U)
    
    epsT = @trace(MappedGenerateEps(fill(hyperparams["tNoise"], n)), :epsT)
    Tcov = broadcast(t_kernel, U, U', hyperparams["utLS"], epsT, epsT', hyperparams["epstLS"])
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    epsY = @trace(MappedGenerateEps(fill(hyperparams["yNoise"], n)), :epsY)
    Ycov = broadcast(y_kernel, U, U', hyperparams["uyLS"], 
                               Tr, Tr', hyperparams["tyLS"], 
                               epsY, epsY', hyperparams["epsyLS"])
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end

@gen (static) function AdditiveNoiseGPROC(hyperparams)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(inv_gamma(hyperparams["utLSShape"], hyperparams["utLSScale"]), :utLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)    
    
#   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    Tcov = broadcast(t_kernel, U, U', utLS, tScale) + tNoise * 1I
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    Ycov = broadcast(y_kernel, U, U', uyLS, Tr, Tr', tyLS, yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end


@gen (static) function LinearAdditiveNoiseGPROC(hyperparams)    
    n = size(hyperparams["SigmaU"])[1]
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(inv_gamma(hyperparams["utLSShape"], hyperparams["utLSScale"]), :utLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS)    
    
#   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    Tcov = broadcast(t_kernel, U, U', utLS, tScale) + tNoise * 1I
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    Ycov = broadcast(y_kernel, U, U', uyLS, Tr, Tr', yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
# -
end
