module ProcessingIHDP

using CSV
using DataFrames
using Random
using LinearAlgebra
using Distributions

export generatePairs, generateSigmaU, generateT, generateX, generateU, generateWeights, generateOutcomes

# +
# rootpath = "IHDP/csv/"
# path(x) = rootpath * "ihdp_npci_$(x).csv"
# makeFrame(x) = DataFrame(CSV.File(path(x)))

# data = makeFrame("1")

# nData = size(data)[1]

# +
# Randomly generate a pair for each instance with probability p
function generatePairs(data, p)
    pairs = randsubseq(1:size(data)[1], p)
end

# n = nData + length(pairs)
# -

# Generate SigmaU
function generateSigmaU(pairs, nData)

    eps = 1e-7

    n = length(pairs) + nData

    SigmaU = Matrix{Float64}(I, n, n)

    for (i, pair) in enumerate(pairs)
        SigmaU[pair, i+nData] = 1.0
    end

    return Symmetric(SigmaU + I * eps)
end

# Generate T
function generateT(data, pairs)
    return vcat(data[!, :T], 1 .- data[pairs, :T])
end

# Generate X by adding noise to original continuous X.
function generateX(data, pairs)

    covariatesObs = [Symbol("x$(i)") for i in 1:6]

    noise = [0.05 * var(data[!, cov]) for cov in covariatesObs]

    X = vcat(data[!, covariatesObs], copy(data[pairs, covariatesObs]))

    for row in 1:size(X)[1]
        for col in 1:length(covariatesObs)
            X[row, col] += rand(Normal(0, noise[col]))
        end
    end

    return X
end

# +
# Generate U - Used only for computing Y

function generateU(data, pairs)
    covariatesLatent = [Symbol("x$(i)") for i in 7:25]
    U = vcat(data[!, covariatesLatent], copy(data[pairs, covariatesLatent]))
    return U
end

# +
# Generate Outcome weight Matrices
weights = [0, 0.1, 0.2, 0.3, 0.4]
weightP = [0.6, 0.1, 0.1, 0.1, 0.1]

function generateWeights(weights, weightsP)
    BetaX = []
    BetaU = []

    covariatesObs = [Symbol("x$(i)") for i in 1:6]
    covariatesLatent = [Symbol("x$(i)") for i in 7:25]

    for i in 1:length(covariatesObs)
        push!(BetaX, weights[rand(Categorical(weightsP))])
    end

    for i in 1:length(covariatesLatent)
        push!(BetaU, weights[rand(Categorical(weightsP))])
    end

    return BetaX, BetaU
end

# +
function outcome(X, U, BetaX, BetaU, omegaB)
    noise = rand(Normal(0, 1))

    mean0 = (exp.(X .+ 0.5)' * BetaX) * (exp.(U .+ 0.5)' * BetaU)
    mean1 = (X' * BetaX) + (U' * BetaU) - omegaB
    return mean0 + noise, mean1 + noise
end

function generateOutcome(X, U, T, BetaX, BetaU, omegaB, n)
    Y = []
    Ycf = []

    for i in 1:n
        outcomePair = outcome(Array(X[i, :]), Array(U[i, :]), BetaX, BetaU, omegaB)
        push!(Y, outcomePair[T[i]+1])
        push!(Ycf, outcomePair[(1-T[i])+1])
    end

    return Y, Ycf
end
# -

# Generate Y
# See Page 11 of "Bayesian Nonparametric Modeling for Causal Inference"
# We use response surface B
function generateOutcomes(X, U, T, BetaX, BetaU, CATT, n)
    Ytmp, Ycftmp = generateOutcome(X, U, T, BetaX, BetaU, 0.0, n)
    treated = T .== 1
    omegaB = mean(Ytmp[treated] - Ycftmp[treated]) - CATT
    Y, Ycf = generateOutcome(X, U, T, BetaX, BetaU, omegaB, n)
    return Y, Ycf
end

end

using .ProcessingIHDP

# from experiment config 9.toml
# because results were added with changes to that file in the same commit

data = DataFrame(CSV.File("IHDP/csv/ihdp_npci_1.csv"))[1:200, :]
pairs = generatePairs(data, 0.3)
nData = size(data)[1]
n = nData + length(pairs)

SigmaU = generateSigmaU(pairs, nData)
T_ = generateT(data, pairs)
T = [Bool(t) for t in T_]
X_ = generateX(data, pairs)
nX = size(X_)[2]
X = [X_[!, i] for i in 1:nX]
U = generateU(data, pairs)
BetaX, BetaU = generateWeights([0.0, 0.1, 0.2, 0.3, 0.4], [0.6, 0.1, 0.1, 0.1, 0.1])
Y_, Ycf = generateOutcomes(X_, U, T_, BetaX, BetaU, 4.0, n)
Y = [Float64(y) for y in Y_]

obj = zeros(size(Y)...)
for i in 1:nData
    obj[i] = i
end
objectIndeces = [[i] for i in 1:nData]
for (i, pair) in enumerate(pairs)
    # println(pair)
    push!(objectIndeces[pair], i + nData)
    obj[i+nData] = pair
end
# println(objectIndeces)

df = DataFrame()
df[!, "T"] = T
# println("T $(size(T))")
for k in 1:nX
    # println("X$k $(size(X[k]))")
    df[!, "X$k"] = X[k]
end
# println("Y $(size(Y))")
df[!, "Y"] = Y
# println("Ycf $(size(Ycf))")
# df[!, "Ycf"] = Ycf
# println("obj $(size(obj))")
df[!, "obj"] = obj

CSV.write("IHDP_sampled.csv", df)