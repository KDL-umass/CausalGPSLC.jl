# +
using Gen
using LinearAlgebra
using PyPlot
using Seaborn

include("../src/inference.jl")
include("../src/estimation.jl")
include("../src/model.jl")
using .Inference
using .Estimation
using .Model

# +
function simQuadraticData(uCov)
    n = size(uCov)[1]
    U = zeros(n)
    X = zeros(n)
    Y = zeros(n)
    epsY = zeros(n)
    
    U = mvnormal(zeros(n), uCov)
    
    for i in 1:n
        X[i] = normal(U[i]+1, 1.0)
        Ymean = (0.2 * X[i] * U[i])
        epsY[i] = normal(0, 0.1)
        Y[i] = Ymean + epsY[i]
    end
    return U, X, Y, epsY
end

function simQuadraticIntData(U, epsY, doX)
    n = length(U)
    
    Yint = zeros(n)
    
    for i in 1:n
        Ymean = (0.2 * doX * U[i])
        Yint[i] = Ymean + epsY[i]
    end
    return Yint
end

# +
# n should be even
n = 10
eps = 0.000000000000001

# All individuals indepedendent.
# uCov = Matrix{Float64}(I, n, n)

# All individuals belong to the same group.
uCov = ones(n, n) + Matrix{Float64}(I, n, n) * eps

# Two individuals per group.
# uCov = Matrix{Float64}(I, n, n) + Matrix{Float64}(I, n, n) * eps
# for i = 1:Integer(n/2)
#     uCov[2*i, 2*i-1] = 1.
#     uCov[2*i-1, 2*i] = 1.
# end

# Two groups with all individuals
# uCov = Matrix{Float64}(I, n, n)
# uCov[1:(Integer(n/2)), 1:(Integer(n/2))] = ones(Integer(n/2), Integer(n/2))
# uCov[Integer(n/2)+1:end, Integer(n/2)+1:end] = ones(Integer(n/2), Integer(n/2))
# uCov += Matrix{Float64}(I, n, n) * eps

println("")


# -

U, X, Y, epsY = simQuadraticData(uCov)
scatter(X, Y, c=U)
colorbar()
title("Color = U")
xlabel("Treatment")
ylabel("Outcome")

# +
# Tune Hyperparameters

hyperparams = Dict()

hyperparams["uxLS"] = 10.
hyperparams["uyLS"] = 10.
hyperparams["xyLS"] = 1.

hyperparams["xNoise"] = 1.
hyperparams["yNoise"] = 0.1

hyperparams["uCov"] = uCov

load_generated_functions()
(tr, _) = generate(AdditiveNoiseGPROC, (hyperparams,))
generatedU = get_choices(tr)[:U]
generatedX = get_choices(tr)[:X]
generatedY = get_choices(tr)[:Y]

scatter(generatedX, generatedY, c=generatedU)
colorbar()
title("Color = Generated U")
xlabel("Generated Treatment")
ylabel("Generated Outcome")
# -

# Inference
nSteps = 20
Us, tr = AdditiveNoisePosterior(hyperparams, X, Y, nSteps)
println("")

# +
# Estimation

burnIn = 1
doX = 1.
_, _, effectMeans, effectVars = SATE([hyperparams for i in burnIn:nSteps], Us[burnIn:end, :], X, Y, doX)

   
nSamplesPerMixture = 1

samples = SATEsamples(effectMeans, effectVars, nSamplesPerMixture)
println("")
# -

Us

function TestYKernel(u1::Float64, u2::Float64, uyLS::Float64, x1::Float64, x2::Float64, xyLS::Float64)
# function y_kernel(u1, u2, uyLS, x1, x2, xyLS)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    return exp(-(u_term + x_term)/2)
end

function TestConditionalSATE(uyLS::Float64, xyLS::Float64, U, X, Y, doX)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i and eps_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovY = broadcast(TestYKernel, U, reshape(U, 1, n), uyLS, X, reshape(X, 1, n), xyLS)
    CovY = Symmetric(CovY)
    println(cond(CovY))
    
#   k_Y,Y_x in the overleaf doc. The cross covariance block is not in general symettric.
    crossCovY = broadcast(TestYKernel, U, reshape(U, 1, n), uyLS, doX, reshape(X, 1, n), xyLS)
    
#   k_Y_x in the overleaf doc.
    intCovY = broadcast(TestYKernel, U, reshape(U, 1, n), uyLS, doX, doX, xyLS)
    intCovY = Symmetric(intCovY)
    
    condMean = crossCovY * (CovY \ Y)
    condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
    effectMean = sum(condMean-Y)/n
    effectVar = sum(condCov)/n
    
    return effectMean, effectVar
end

# +
index = 20

TestConditionalSATE(hyperparams["uyLS"], hyperparams["xyLS"], Us[index, :], X, Y, doX)

# +
intY = simQuadraticIntData(U, epsY, doX)

# kdeplot(intY - Y, label="(Y|do(X=$doX)) - Y", c="r")
axvline(sum(intY - Y)/n, c="r", ymax=0.1, label="Ground Truth")
kdeplot(samples, label="GPROC Estimate")

# kdeplot(intY, label="(Y|do(X=$doX))", c="b")
# axvline(sum(intY1)/n, c="b", ymax=0.1)

# kdeplot(Y, label="Y", c="black")
# axvline(sum(Y)/n, c="black", ymax=0.1)


# hist(intY, label="(Y|do(X=$doX))")

legend()
xlabel("SATE")
ylabel("P(SATE)")
# -


