using Gen
using PyPlot
using LinearAlgebra
import Base.show

# +
function x_kernel(u1, u2, uxLS)
    return exp(-(((u1 - u2)/uxLS)^2)/2)
end

function y_kernel(u1, u2, uyLS, x1, x2, xyLS)
    u_term = ((u1 - u2)/uyLS)^2
    x_term = ((x1 - x2)/xyLS)^2
    
    return exp(-(u_term + x_term)/2)
end
# -

@gen (grad) function generateU(covU::Array{2, Float64})
    n = size(covU)[1]
    U = @trace(mvnormal(zeros(n), covU), :U)
    return U
end

@gen (grad) function generateX(uxLS::Float64, noise::Float64, (grad)(U))
    n = length(U)
    Xcov = zeros(n, n)
    for i in 1:n
        for j in i:n
            Xcov[i, j] = x_kernel(U[i], U[j], uxLS)
        end
    end
    
    Xcov = Symmetric(Xcov + noise * 1I)
#     println(Xcov)
    X = @trace(mvnormal(zeros(n), Xcov), :X)
    
    return X
end


@gen (grad) function generateY(uyLS::Float64, xyLS::Float64, noise::Float64, (grad)(U), (grad)(X))
    n = length(U)
    Ycov = zeros(n, n)
    for i in 1:n
        for j in 1:n
            Ycov[i, j] = y_kernel(U[i], U[j], uyLS, X[i], X[j], xyLS)
        end
    end
    
    Ycov = Symmetric(Ycov + noise * 1I)
    
    Y = @trace(mvnormal(zeros(n), Ycov), :Y)

    return Y
end

@gen (grad) function GPROC(hyperparams)    
    U = @trace(generateU(hyperparams["uCov"]))
    X = @trace(generateX(hyperparams["uxLS"], hyperparams["xNoise"], U))
    Y = @trace(generateY(hyperparams["uyLS"], hyperparams["xyLS"], hyperparams["yNoise"], U, X))
end

# +
@gen function uProposal(trace, uCov, indeces, n) 
    cov = Matrix{Float64}(I, n, n) * eps
    cov[indeces, indeces] = uCov[indeces, indeces]
    U = @trace(mvnormal(get_choices(trace)[:U], cov), :U)
end

function inference(hyperparams, X, Y, blocks, nSteps)
    obs = Gen.choicemap()
    obs[:X] = X
    obs[:Y] = Y
    
    n = length(X)
    
    (tr, _) = generate(GPROC, (hyperparams,), obs)
    for iter=1:nSteps
        for indeces in blocks 
            (tr, _) = mh(tr, uProposal, (hyperparams["uCov"], indeces, n))
        end
        
        (tr, _) = mala(tr, select(:U), 0.1)
    end
    tr
end
# -

function conditionalSATE(uyLS::Float64, xyLS::Float64, noise::Float64, U, X, Y, doX)
#   Generate a new post-intervention instance for each data instance in
#   the dataset. This data instance has the same U_i, but X[i] is replaced
#   with doX.
    
#   This assumes that the confounder U and kernel hyperparameters are known. 
#   To compute the SATE marginalized over P(U, lambda|X, Y) this function can
#   be used to compute monte carlo estimates.
    
    n = length(U)
    
    CovY = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            CovY[i, j] = y_kernel(U[i], U[j], uyLS, X[i], X[j], xyLS)
        end
    end
    
    CovY = Symmetric(CovY + noise * 1I)
    
#   k_Y,Y_x in the overleaf doc.
    crossCovY = zeros(n, n)
    
#   k_Y_x in the overleaf doc.
    intCovY = zeros(n, n)
    
    for i in 1:n
        for j in 1:n
            crossCovY[i, j] = y_kernel(U[i], U[j], uyLS, doX, X[j], xyLS)
            intCovY[i, j] = y_kernel(U[i], U[j], uyLS, doX, doX, xyLS)
        end
    end
    
#   The cross covariance block is not in general symettric.
    intCovY = Symmetric(intCovY + noise * 1I)
    
    condMean = crossCovY * (CovY \ Y)
    condCov = intCovY - (crossCovY * (CovY \ transpose(crossCovY)))
    effectMean = sum(condMean-Y)/n
    effectVar = sum(condCov)/n
    
    return effectMean, effectVar
end

function SATE(postHyp, postU, X, Y, doX)
    
    nPostSamples = length(postU)
    
    effectMeans = zeros(nPostSamples)
    effectVars = zeros(nPostSamples)
    
    for i in 1:nPostSamples
#         effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["xyLS"], postHyp[i]["yNoise"], 
#                                                         postU[i], X, Y, doX)
         effectMeans[i], effectVars[i] = conditionalSATE(postHyp[i]["uyLS"], postHyp[i]["xyLS"], 0.00001, 
                                                         postU[i], X, Y, doX)
    end
    
    n = length(effectMeans)
    
    totalMean = sum(effectMeans)/n
    totalVar = sum(effectVars)/n + sum(effectMeans .* effectMeans)/n - ((sum(effectMeans))^2)/n^2
    
    return totalMean, totalVar
end

# # DEMO

# +
# n Should be even.
n = 30

function simData(n)
    U = zeros(n)
    X = zeros(n)
    Y = zeros(n)
    IntY = zeros(n)
    
    for i in 1:n
        U[i] = uniform(-3, 3)
        X[i] = normal(U[i], 1)
        Ymean = (0.3 - 0.1 * X[i] + 0.5 * X[i]^2 - 1 * U[i])/5
        Y[i] = normal(Ymean, 0.1)
    end
    return U, X, Y
end

U, X, Y = simData(n)
scatter(X, Y, c=U)

# +
eps = 0.0000001

hyperparams = Dict()

hyperparams["uxLS"] = 0.5
hyperparams["uyLS"] = 0.5
hyperparams["xyLS"] = 1.

hyperparams["xNoise"] = 1.
hyperparams["yNoise"] = 0.5

# All individuals indepedendent.
# uCov = Matrix{Float64}(I, n, n)
# blocks = [[i] for i in 1:n]

# All individuals belong to the same group.
# uCov = ones(n, n) + Matrix{Float64}(I, n, n) * eps
# blocks = [[i for i in 1:n]]

# Two individuals per group.
uCov = Matrix{Float64}(I, n, n) + Matrix{Float64}(I, n, n) * eps
for i = 1:Integer(n/2)
    uCov[2*i, 2*i-1] = 1.
    uCov[2*i-1, 2*i] = 1.
end
blocks = [[2*i-1, 2*i] for i in 1:Integer(n/2)]

# Two groups with all individuals
# uCov = Matrix{Float64}(I, n, n)
# uCov[1:(Integer(n/2)), 1:(Integer(n/2))] = ones(Integer(n/2), Integer(n/2))
# uCov[Integer(n/2)+1:end, Integer(n/2)+1:end] = ones(Integer(n/2), Integer(n/2))
# uCov += Matrix{Float64}(I, n, n) * eps
# blocks = [[i for i in 1:Integer(n/2)], [i for i in Integer(n/2)+1:n]]

hyperparams["uCov"] = Symmetric(uCov)
# (trace, _) = generate(GPROC, (hyperparams,))

nSteps = 100
nSamples = 10

uPosteriorSamples = [get_choices(inference(hyperparams, X, Y, blocks, nSteps))[:U] for i in 1:nSamples]
nSteps
# -

uPosteriorSamples[1][1:6]

uPosteriorSamples[2][1:6]

postMean, postVar = SATE([hyperparams for i in 1:length(uPosteriorSamples)], uPosteriorSamples, X, Y, 0.)











# # Scratch Work Below

@gen function conditional_normal_params(X, cov, i)
    
    mean = transpose(cov[i, 1:i-1]) * ((cov[1:i-1, 1:i-1]) \ X[1:i-1])
    var = cov[i, i] - (transpose(cov[i, 1:i-1]) * (cov[1:i-1, 1:i-1] \ cov[i, 1:i-1]))
    
    return mean, var
end

function particle_filter(num_particles::Int, hyperparams, X, Y, num_samples)
    state = Gen.initialize_particle_filter(GPROC, (0, hyperparams), choicemap(), num_particles)
    
    obs = Gen.choicemap()
    for current_n in 1:length(X)
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
        
        obs[(:X, current_n)] = X[current_n]
        obs[(:Y, current_n)] = Y[current_n]
        println(obs)
        
        Gen.particle_filter_step!(state, (current_n, hyperparams), (Gen.UnknownChange, Gen.NoChange), obs)
    end

    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples), state
end;

particle_filter(100, hyperparams, X, Y, 10)

# +
n_samples = 10


U = generateU(n_samples)
X = generateX(n_samples, hyperparams["uxLS"], hyperparams["xNoise"], U)
Y = generateY(n_samples, hyperparams["uyLS"], hyperparams["xyLS"], hyperparams["yNoise"], U, X)

# +
n_curves = 3
n_samples = 10

uxLS = 0.5
uyLS = 1.5
xyLS = 1.5

xNoise = 0.3
yNoise = 0.0001

U = generateU(n_samples)

for i in 1:n_curves
    X = generateX(n_samples, uxLS, xNoise, U)
    Y = generateY(n_samples, uyLS, xyLS, yNoise, U, X)
    scatter(X, Y, s=U*50)
end

plt.xlabel("X")
plt.ylabel("Y")

# +
fig, axes = plt.subplots(3,3, figsize=(10, 10))

Umin = 10000
Umax = -10000
Xmin = 10000
Xmax = -10000
Ymin = 10000
Ymax = -10000

for _ in 1:n_curves
    U = generateU(n_samples)
    X = generateX(n_samples, uxLS, epsilon, U)
    Y = generateY(n_samples, uyLS, xyLS, epsilon, U, X)
    
    Umin = minimum([Umin, minimum(U)])
    Umax = maximum([Umax, maximum(U)])
    Xmin = minimum([Xmin, minimum(X)])
    Xmax = maximum([Xmax, maximum(X)])
    Ymin = minimum([Ymin, minimum(Y)])
    Ymax = maximum([Ymax, maximum(Y)])
    

    key = Dict([(1, U), (2, X), (3, Y)])
    key_string = Dict([(1, "U"), (2, "X"), (3, "Y")])
    key_min = Dict([(1, Umin), (2, Xmin), (3, Ymin)])
    key_max = Dict([(1, Umax), (2, Xmax), (3, Ymax)])
    
    for i in 1:3
        for j in 1:3
            axes[i, j].scatter(key[i], key[j])
            axes[i, j].set_xlim(key_min[i], key_max[i])
            axes[i, j].set_ylim(key_min[j], key_max[j])
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])

            end
        axes[i, 1].set_ylabel(key_string[i])
        axes[1, i].set_title(key_string[i])
    end
end
# -

minimum([1,2,3])


