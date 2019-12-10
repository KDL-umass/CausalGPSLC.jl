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
function simLinearData(SigmaU, tNoise, yNoise, uNoise, UTslope, UYslope, TYslope)
    n = size(SigmaU)[1]
    U = zeros(n)
    T = zeros(n)
    Y = zeros(n)
    epsY = zeros(n)
    
    U = mvnormal(zeros(n), SigmaU * uNoise)
    
    for i in 1:n
        T[i] = normal(U[i] * UTslope, tNoise)
        Ymean = (T[i] * TYslope + U[i] * UYslope)
        epsY[i] = normal(0, yNoise)
        Y[i] = Ymean + epsY[i]
    end
    return U, T, Y, epsY
end

function simLinearIntData(U, epsY, doT, TYslope)
    n = length(U)
    
    Yint = zeros(n)
    
    for i in 1:n
        Ymean = (TYslope * doT)
        Yint[i] = Ymean + epsY[i]
    end
    return Yint
end

# +
# n should be even
n = 10
eps = 0.0000000000001
uNoise = .5
tNoise = 1.
yNoise = 0.1

UTslope = 1.
UYslope = -1.
TYslope = 1.


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

U, T, Y, epsY = simLinearData(SigmaU, tNoise, yNoise, uNoise, UTslope, UYslope, TYslope)
scatter(T, Y, c=U)
colorbar()
title("Color = U")
xlabel("Treatment")
ylabel("Outcome")

# +
# Set Hyperparameters

hyperparams = Dict()

hyperparams["tNoiseMin"], hyperparams["tNoiseMax"] = 0.01, 3.
hyperparams["yNoiseMin"], hyperparams["yNoiseMax"] = 0.01, 3.
hyperparams["uNoiseMin"], hyperparams["uNoiseMax"] = 0.01, 3.

hyperparams["utLSShape"], hyperparams["utLSScale"] = 4., 4.
hyperparams["uyLSShape"], hyperparams["uyLSScale"] = 4., 4.
hyperparams["tyLSShape"], hyperparams["tyLSScale"] = 4., 4.

hyperparams["tScaleShape"], hyperparams["tScaleScale"] = 4., 4.
hyperparams["yScaleShape"], hyperparams["yScaleScale"] = 4., 4.

hyperparams["SigmaU"] = SigmaU
load_generated_functions()
(trace, _) = generate(AdditiveNoiseGPROC, (hyperparams,))
generatedU = get_choices(trace)[:U]
generatedT = get_choices(trace)[:Tr]
generatedY = get_choices(trace)[:Y]

scatter(generatedT, generatedY, c=generatedU)
colorbar()
title("Color = Generated U")
xlabel("Generated Treatment")
ylabel("Generated Outcome")
# -

# Inference
nOuter = 1000
nMHInner = 1
nESInner = 1
PosteriorSamples, trace = AdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
println("")

# +
# Estimation
burnIn = 100
MHStep = 10
doT = -2.
MeanSATEs, VarSATEs = SATE(PosteriorSamples[burnIn:MHStep:end], T, Y, doT)

nSamplesPerMixture = 100

samples = SATEsamples(MeanSATEs, VarSATEs, nSamplesPerMixture)
println("")

# +
intY = simLinearIntData(U, epsY, doT, TYslope)

# kdeplot(intY - Y, label="(Y|do(X=$doX)) - Y", c="r")
axvline(sum(intY - Y)/n, c="r", ymax=0.1, label="Ground Truth")
kdeplot(samples, label="GPROC Estimate")

legend()
xlabel("SATE")
ylabel("P(SATE)")
# -


