module Model

# +
using Gen
using LinearAlgebra
import Base.show
import FunctionalCollections

export t_kernel, y_kernel, AdditiveNoiseGPROC, LinearAdditiveNoiseGPROC

# +
# Lot's of overloaded functions below.


# RBF kernel over U, eps, and X
function t_kernel(u1::Float64, u2::Float64, utLS::Float64, eps1::Float64, eps2::Float64, 
                  epstLS::Float64, xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, 
                  xtLS, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    eps_term = ((eps1 - eps2)/epstLS)^2
    x_term =  (((xpair[1] - xpair[2])./xtLS).^2)
    return tScale * exp(-(u_term + eps_term + sum(x_term))/2)
end

# RBF kernel over U and eps. No X.
function t_kernel(u1::Float64, u2::Float64, utLS::Float64, eps1::Float64, eps2::Float64, 
                  epstLS::Float64, xpair::Nothing, 
                  xtLS, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    eps_term = ((eps1 - eps2)/epstLS)^2
    return tScale * exp(-(u_term + eps_term)/2)
end

# RBF kernel over U and X. Additive eps handled by adding diagonal to kernel in model.
function t_kernel(u1::Float64, u2::Float64, utLS::Float64, xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, 
                  xtLS, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    x_term = (((xpair[1] - xpair[2])./xtLS).^2)
    return tScale * exp(-(u_term + sum(x_term))/2)
end

# RBF kernel over U. Additive eps handled by adding diagonal to kernel in model.
function t_kernel(u1::Float64, u2::Float64, utLS::Float64, xpair::Nothing, 
                  xtLS, tScale::Float64)
    u_term = ((u1 - u2)/utLS)^2
    return tScale * exp(-u_term/2)
end

# RBF kernel over U, T, eps, and X.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64,
                  tyLS::Float64, eps1::Float64, eps2::Float64, epsyLS::Float64, 
                  xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, 
                  xyLS, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    eps_term = ((eps1 - eps2)/epsyLS)^2
    x_term = (((xpair[1] - xpair[2])./xyLS).^2)
    return yScale * exp(-(u_term + t_term + eps_term + sum(x_term))/2)
end

# RBF kernel over U, T, and eps. No X.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64,
                  tyLS::Float64, eps1::Float64, eps2::Float64, epsyLS::Float64, 
                  xpair::Nothing, xyLS, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    eps_term = ((eps1 - eps2)/epsyLS)^2
    return yScale * exp(-(u_term + t_term + eps_term)/2)
end

# RBF kernel over U, T, and X. Additive eps handled by adding diagonal to kernel in model.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64, 
                  tyLS::Float64, xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, 
                  xyLS, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    x_term = (((xpair[1] - xpair[2])./xyLS).^2)
    return yScale * exp(-(u_term + t_term + sum(x_term))/2)
end

# RBF kernel over U and T. Additive eps handled by adding diagonal to kernel in model. No X.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64, 
                  tyLS::Float64, xpair::Nothing, xyLS, 
                  yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = ((t1 - t2)/tyLS)^2
    return yScale * exp(-(u_term + t_term)/2)
end

# RBF kernel over U and X, linear kernel over T. Additive eps handled by adding diagonal to kernel in model.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64,
                  xpair::Tuple{Array{Float64, 1}, Array{Float64, 1}}, 
                  xyLS, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = t1 * t2
    x_term = (((xpair[1] - xpair[2])./xyLS).^2)

    return yScale * (exp(-(u_term + sum(x_term))/2) + t_term)
end

# RBF kernel over U and linear kernel over T. Additive eps handled by adding diagonal to kernel in model. No X.
function y_kernel(u1::Float64, u2::Float64, uyLS::Float64, t1::Float64, t2::Float64,
                  xpair::Nothing, xyLS, yScale::Float64)
    u_term = ((u1 - u2)/uyLS)^2
    t_term = t1 * t2

    return yScale * (exp(-u_term/2) + t_term)
end

# +
# @gen (grad, static) function generateEps(noise::Float64)
#     eps = @trace(normal(0, noise), :eps)
#     return eps
# end

# MappedGenerateEps = Map(generateEps)

# @gen (static) function GPROC(hyperparams)    
#     n = size(hyperparams["SigmaU"])[1]
    
#     U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"]), :U)
    
#     epsT = @trace(MappedGenerateEps(fill(hyperparams["tNoise"], n)), :epsT)
#     Tcov = broadcast(t_kernel, U, U', hyperparams["utLS"], epsT, epsT', hyperparams["epstLS"])
#     Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
#     epsY = @trace(MappedGenerateEps(fill(hyperparams["yNoise"], n)), :epsY)
#     Ycov = broadcast(y_kernel, U, U', hyperparams["uyLS"], 
#                                Tr, Tr', hyperparams["tyLS"], 
#                                epsY, epsY', hyperparams["epsyLS"])
#     Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
#     return Y
# end

@gen (static) function generate_LS(shape, scale)
    LS = @trace(inv_gamma(shape, scale), :LS)
    return LS
end

MappedGenerateLS = Map(generate_LS)

load_generated_functions()

@gen (static) function AdditiveNoiseGPROC(hyperparams, Xcol, nX)    
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
    
#     Tcov = broadcast(t_kernel, U, U', utLS, tScale) + tNoise * 1I
    Tcov = broadcast(t_kernel, U, U', utLS, Xcol, (xtLS,), tScale) + tNoise * 1I
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    Ycov = broadcast(y_kernel, U, U', uyLS, Tr, Tr', tyLS,  Xcol, (xyLS,), yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end


@gen (static) function LinearAdditiveNoiseGPROC(hyperparams, Xcol, nX)    
    n = size(hyperparams["SigmaU"])[1]
    
    
#   Prior over Noise
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    
#   Prior over Kernel Lengthscales
    utLS = @trace(inv_gamma(hyperparams["utLSShape"], hyperparams["utLSScale"]), :utLS)
    uyLS = @trace(inv_gamma(hyperparams["uyLSShape"], hyperparams["uyLSScale"]), :uyLS) 
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX), 
                                      fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX), 
                                      fill(hyperparams["xyLSScale"], nX)), :xyLS)
    
#   Prior over Kernel Scale
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
  
#   Generate Data 
    U = @trace(mvnormal(fill(0, n), hyperparams["SigmaU"] * uNoise), :U)
    
    Tcov = broadcast(t_kernel, U, U', utLS, Xcol, (xtLS,), tScale) + tNoise * 1I
    Tr = @trace(mvnormal(fill(0, n), Tcov), :Tr)
    
    Ycov = broadcast(y_kernel, U, U', uyLS, Tr, Tr', Xcol, (xyLS,), yScale) + yNoise * 1I
    Y = @trace(mvnormal(fill(0, n), Ycov), :Y)
    return Y
end
# -
end
