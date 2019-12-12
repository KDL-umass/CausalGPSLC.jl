# +
using Gen
using LinearAlgebra
using PyPlot
using Seaborn
using StatsBase

include("../src/inference.jl")
include("../src/estimation.jl")
include("../src/model.jl")
using .Inference
using .Estimation
using .Model
# -

# # Synthetic Data Generators

function generateSigmaU(n::Int, nIndividualsArray::Array{Int}, eps::Float64, cov::Float64)
    SigmaU = Matrix{Float64}(I, n, n)
    
    i = 1
    for nIndividuals in nIndividualsArray
        SigmaU[i:i+nIndividuals-1,i:i+nIndividuals-1] = ones(nIndividuals, nIndividuals) * cov
        i+= nIndividuals
    end
    
    SigmaU[diagind(SigmaU)] .= 1 + eps
    
    return SigmaU + Matrix{Float64}(I, n, n) * eps
end

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

function simLinearIntData(U, epsY, doT, TYslope, UYslope)
    n = length(epsY)
    
    Yint = zeros(n)
    
    for i in 1:n
        Ymean = (doT * TYslope + U[i] * UYslope)
        Yint[i] = Ymean + epsY[i]
    end
    return Yint
end

# +
# n should be even
n = 100
eps = 0.0000000000001
uNoise = 0.2
tNoise = 1.
yNoise = 0.1

UTslope = 2.
UYslope = 2.
TYslope = -0.5


SigmaU = generateSigmaU(n, [10 for i in 1:n/10], eps, 1.)

# SigmaU = Matrix{Float64}(I, n, n)
println()
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

hyperparams["tNoiseShape"], hyperparams["tNoiseScale"] = 4., 4.
hyperparams["yNoiseShape"], hyperparams["yNoiseScale"] = 4., 4.
hyperparams["uNoiseShape"], hyperparams["uNoiseScale"] = 4., 4.

hyperparams["utLSShape"], hyperparams["utLSScale"] = 4., 4.
hyperparams["uyLSShape"], hyperparams["uyLSScale"] = 4., 4.
hyperparams["tyLSShape"], hyperparams["tyLSScale"] = 4., 4.

hyperparams["tScaleShape"], hyperparams["tScaleScale"] = 4., 4.
hyperparams["yScaleShape"], hyperparams["yScaleScale"] = 4., 4.

hyperparams["SigmaU"] = SigmaU
load_generated_functions()
(trace, _) = generate(AdditiveNoiseGPROC, (hyperparams,))
println(get_choices(trace)[:uNoise])
generatedU = get_choices(trace)[:U]
generatedT = get_choices(trace)[:Tr]
generatedY = get_choices(trace)[:Y]

scatter(generatedT, generatedY, c=generatedU)
colorbar()
title("Color = Generated U")
xlabel("Generated Treatment")
ylabel("Generated Outcome")
# -

# # Dose - Response Curve

# +
nOuter = 1000
nMHInner = 1
nESInner = 5
burnIn = 100
MHStep = 25
doTs = collect(range(quantile(T, 0.1), length=100, stop=quantile(T, 0.9)))
LinearSamples = []
RBFSamples = []
truths = []
nSamplesPerMixture = 100

Ymean = sum(Y)/length(Y)

# LinearPosteriorSamples, _ = LinearAdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
for doT in doTs
    LinearMeanSATEs, LinearVarSATEs = LinearSATE(LinearPosteriorSamples[burnIn:MHStep:end], T, Y, doT)
    push!(LinearSamples, SATEsamples(LinearMeanSATEs, LinearVarSATEs, nSamplesPerMixture) .+ Ymean)
end

# PosteriorSamples, _ = AdditiveNoisePosterior(hyperparams, T, Y, nOuter, nMHInner, nESInner)
for doT in doTs
    MeanSATEs, VarSATEs = SATE(PosteriorSamples[burnIn:MHStep:end], T, Y, doT)
    push!(RBFSamples, SATEsamples(MeanSATEs, VarSATEs, nSamplesPerMixture) .+ Ymean)
    push!(truths, mean(simLinearIntData(U, epsY, doT, TYslope, UYslope)))
end

println()

# +
X = zeros(length(T),2)
X[:,1] = T  
X[:,2] = ones(length(T))

Reg = X\Y

# +
LinearGPROCmeans = [mean(sample) for sample in LinearSamples]
LinearGPROCuppers = [quantile(sample, 0.9) for sample in LinearSamples]
LinearGPROClowers = [quantile(sample, 0.1) for sample in LinearSamples]

RBFGPROCmeans = [mean(sample) for sample in RBFSamples]
RBFGPROCuppers = [quantile(sample, 0.9) for sample in RBFSamples]
RBFGPROClowers = [quantile(sample, 0.1) for sample in RBFSamples]

linewidth = 1

plot(doTs, LinearGPROCmeans, c="red", linewidth=linewidth, linestyle="--")
plot(doTs, LinearGPROClowers, label="GPROC w/ Linear Kernel E[Y|do(T=T)]", c="red", linewidth=linewidth)
plot(doTs, LinearGPROCuppers, c="red", linewidth=linewidth)

plot(doTs, RBFGPROCmeans, c="blue", linewidth=linewidth, linestyle="--")
plot(doTs, RBFGPROClowers, label="GPROC w/ RBF Kernel E[Y|do(T=T)]", c="blue", linewidth=linewidth)
plot(doTs, RBFGPROCuppers, c="blue", linewidth=linewidth)

plot(doTs, (Reg[1] .* doTs) .+ Reg[2], label="OLS Regression E[Y|T]", c="green")
scatter(T, Y, c=U, label="Observational Data")
plot(doTs, truths, label="Ground Truth", c="black")


xlim(quantile(T, 0.1), quantile(T, 0.9))
ylim(quantile(Y, 0.1), quantile(Y, 0.9))
xlabel("T")
ylabel("Y")
legend()
# -


