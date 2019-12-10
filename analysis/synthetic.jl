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
# -

# # Synthetic Data Generators

# +
function simLinearData(SigmaU, xNoise, yNoise, uNoise, UXslope, UYslope, XYslope)
    n = size(SigmaU)[1]
    U = zeros(n)
    X = zeros(n)
    Y = zeros(n)
    epsY = zeros(n)
    
    U = mvnormal(zeros(n), SigmaU * uNoise)
    
    for i in 1:n
        X[i] = normal(U[i] * UXslope, xNoise)
        Ymean = (X[i] * XYslope + U[i] * UYslope)
        epsY[i] = normal(0, yNoise)
        Y[i] = Ymean + epsY[i]
    end
    return U, X, Y, epsY
end

function simLinearIntData(U, epsY, doX, XYslope)
    n = length(U)
    
    Yint = zeros(n)
    
    for i in 1:n
        Ymean = (XYslope * doX)
        Yint[i] = Ymean + epsY[i]
    end
    return Yint
end

# +
# n should be even
n = 100
eps = 0.0000000000001
uNoise = .5
xNoise = 1.
yNoise = 0.1


UXslope = 1.
UYslope = -1.
XYslope = 1.


# All individuals indepedendent.
# SigmaU = Matrix{Float64}(I, n, n)

# All individuals belong to the same group.
# SigmaU = ones(n, n) + Matrix{Float64}(I, n, n) * eps

# Two individuals per group.
SigmaU = Matrix{Float64}(I, n, n) + Matrix{Float64}(I, n, n) * eps
for i = 1:Integer(n/2)
    SigmaU[2*i, 2*i-1] = 1.
    SigmaU[2*i-1, 2*i] = 1.
end

# Two groups with all individuals
# SigmaU = Matrix{Float64}(I, n, n)
# SigmaU[1:(Integer(n/2)), 1:(Integer(n/2))] = ones(Integer(n/2), Integer(n/2))
# SigmaU[Integer(n/2)+1:end, Integer(n/2)+1:end] = ones(Integer(n/2), Integer(n/2))
# SigmaU += Matrix{Float64}(I, n, n) * eps

println("")
# -

U, X, Y, epsY = simLinearData(SigmaU, xNoise, yNoise, uNoise, UXslope, UYslope, XYslope)
scatter(X, Y, c=U)
colorbar()
title("Color = U")
xlabel("Treatment")
ylabel("Outcome")

# +
# Set Hyperparameters

hyperparams = Dict()

hyperparams["xNoiseMin"], hyperparams["xNoiseMax"] = 0.01, 3.
hyperparams["yNoiseMin"], hyperparams["yNoiseMax"] = 0.01, 3.
hyperparams["uNoiseMin"], hyperparams["uNoiseMax"] = 0.01, 3.

hyperparams["uxLSShape"], hyperparams["uxLSScale"] = 4., 4.
hyperparams["uyLSShape"], hyperparams["uyLSScale"] = 4., 4.
hyperparams["xyLSShape"], hyperparams["xyLSScale"] = 4., 4.

hyperparams["xScaleShape"], hyperparams["xScaleScale"] = 4., 4.
hyperparams["yScaleShape"], hyperparams["yScaleScale"] = 4., 4.

hyperparams["SigmaU"] = SigmaU

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
nOuter = 1000
nMHInner = 1
nESInner = 1
PosteriorSamples, tr = AdditiveNoisePosterior(hyperparams, X, Y, nOuter, nMHInner, nESInner)
println("")

# +
# Estimation
burnIn = 100
MHStep = 10
doX = -2.
MeanSATEs, VarSATEs = SATE(PosteriorSamples[burnIn:MHStep:end], X, Y, doX)

nSamplesPerMixture = 100

samples = SATEsamples(MeanSATEs, VarSATEs, nSamplesPerMixture)
println("")

# +
intY = simLinearIntData(U, epsY, doX, XYslope)

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
