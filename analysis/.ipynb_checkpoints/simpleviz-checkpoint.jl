# +
using Gen
using LinearAlgebra
using PyPlot

include("../src/model.jl")
include("../src/estimation.jl")
using .Model
using .Estimation

# +
obs = Gen.choicemap()
obs[:X] = [0., 1.]
obs[:Y] = [0., 1.]

priorObs = Gen.choicemap()

doX = 1.

nPartitions = 100
Umax = 3
Umin = -3

hyperparams = Dict()

hyperparams["uxLS"] = 1.
hyperparams["uyLS"] = 1.
hyperparams["xyLS"] = 1.

hyperparams["xNoise"] = 0.05
hyperparams["yNoise"] = 0.05

hyperparams["uCov"] = Matrix{Float64}(I, 2, 2)
# hyperparams["uCov"][1, 2] = 0.8
# hyperparams["uCov"][2, 1] = 0.8

# +
nObs = size(obs[:X])[1]

k = 0 

data = zeros((nPartitions+1)^2, 7)

for i in 0:nPartitions
    for j in 0:nPartitions
        k += 1
        U = [((i/nPartitions) * (Umax - Umin)) + Umin, ((j/nPartitions) * (Umax - Umin)) + Umin]

        obs[:U] = U
        priorObs[:U] = U
        
        _, jointWeight = generate(AdditiveNoiseGPROC, (hyperparams,), obs)
        
        _, priorWeight = generate(AdditiveNoiseGPROC, (hyperparams,), priorObs)
        
        likelihoodWeight = jointWeight - priorWeight 
        
        condSATEmean, condSATEvar = conditionalSATE(hyperparams["uxLS"], hyperparams["xyLS"], 
                                                    U, obs[:X], obs[:Y], doX)
        
        data[k, :] = [U[1], U[2], jointWeight, priorWeight, likelihoodWeight, condSATEmean, condSATEvar]
    end
end
# -

scatter(data[:, 1], data[:, 2], c=exp.(data[:, 4]))
xlabel("U1")
ylabel("U2")
title("P(U1, U2)")
colorbar()

scatter(data[:, 1], data[:, 2], c=exp.(data[:, 5]))
xlabel("U1")
ylabel("U2")
title("P(X1=$(Int(obs[:X][1])), X2=$(obs[:X][2]), Y1=$(Int(obs[:Y][1])), Y2=$(Int(obs[:Y][2]))|U1, U2)")
colorbar()

scatter(data[:, 1], data[:, 2], c=exp.(data[:, 3]))
xlabel("U1")
ylabel("U2")
title("P(U1, U2, X1=$(obs[:X][1]), X2=$(obs[:X][2]), Y1=$(obs[:Y][1]), Y2=$(obs[:Y][2]))")
colorbar()

scatter(data[:, 1], data[:, 2], c=data[:, 6])
xlabel("U1")
ylabel("U2")
title("E[SATE|X1=$(Int(obs[:X][1])), X2=$(Int(obs[:X][2])), Y1=$(Int(obs[:Y][1])), Y2=$(Int(obs[:Y][2])), U1, U2]")
colorbar()

scatter(data[:, 1], data[:, 2], c=data[:, 7])
xlabel("U1")
ylabel("U2")
title("Var[SATE|X1=$(Int(obs[:X][1])), X2=$(Int(obs[:X][2])), Y1=$(Int(obs[:Y][1])), Y2=$(Int(obs[:Y][2])), U1, U2]")
colorbar()

nObs = size(obs[:X])[1]
nSamples = 10000
burnIn = 3000

postU, tr = samplePosterior(hyperparams, obs[:X], obs[:Y], nSamples)
meanSATE, varSATE = SATE([hyperparams for i in 1:nSamples], postU, obs[:X], obs[:Y], doX)

scatter(postU[burnIn:end, 1], postU[burnIn:end, 2], c=[i for i in burnIn:nSamples], s=0.1)
xlabel("U1")
ylabel("U2")
colorbar()
