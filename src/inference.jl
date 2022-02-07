module Inference

using Gen
using ProgressBars

include("model.jl")
using .Model

include("proposal.jl")
using .Proposal

import Base.show
export Posterior

load_generated_functions()

"""Full Model"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Float64}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)
    nX = length(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    for i in 1:nX
        obs[:X=>i=>:X] = X[i]
    end

    PosteriorSamples = []

    (trace, _) = generate(ContinuousGPSLC, (hyperparams, nX, nU), obs)
    for i in tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k::Int = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i = k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uxLS", i = k, j = l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xNoise", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xScale", i = k)))
            end

            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))
        end

        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]

        for j = 1:nESInner
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

"""No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Float64}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    PosteriorSamples = []

    (trace, _) = generate(NoCovContinuousGPSLC, (hyperparams, nU), obs)
    for i in tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise"),))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i = k)))
            end

            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))
        end

        uCov = hyperparams["SigmaU"] * get_choices(trace)[:uNoise]

        for j = 1:nESInner
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

"""No latent confounder"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Float64}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)

    n = length(T)
    nX = length(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    PosteriorSamples = []

    (trace, _) = generate(NoUContinuousGPSLC, (hyperparams, X), obs)
    for i = 1:nOuter
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

        for k = 1:nX
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i = k)))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i = k)))
        end

        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

"""No Covariates or latent confounders"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Float64}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)

    n = length(T)

    obs = Gen.choicemap()
    obs[:Y] = Y

    PosteriorSamples = []

    (trace, _) = generate(NoCovNoUContinuousGPSLC, (hyperparams, T), obs)
    for i = 1:nOuter
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end


"""Binary Treatment Full Model"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Bool}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)
    nX = length(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    for i in 1:nX
        obs[:X=>i=>:X] = X[i]
    end

    PosteriorSamples = []

    (trace, _) = generate(BinaryGPSLC, (hyperparams, nX, nU), obs)
    for i in tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i = k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uxLS", i = k, j = l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xNoise", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xScale", i = k)))
            end

            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        U = [choices[:U=>i=>:U] for i in 1:nU]
        utLS = [choices[:utLS=>i=>:LS] for i in 1:nU]
        xtLS = [choices[:xtLS=>i=>:LS] for i in 1:nX]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]

        utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
        xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
        logitTcov = processCov(utCovLog + xtCovLog, tScale, tNoise)

        uCov = hyperparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTcov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

"""Binary Treatment No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Bool}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    PosteriorSamples = []

    (trace, _) = generate(NoCovBinaryGPSLC, (hyperparams, nU), obs)
    for i = 1:nOuter
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i = k)))
            end

            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        U = [choices[:U=>i=>:U] for i in 1:nU]
        utLS = [choices[:utLS=>i=>:LS] for i in 1:nU]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]

        utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
        logitTcov = processCov(utCovLog, tScale, tNoise)

        uCov = hyperparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTcov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

# Binary Treatment No Confounders
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Bool}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = length(T)
    nX = length(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    PosteriorSamples = []

    (trace, _) = generate(NoUBinaryGPSLC, (hyperparams, X), obs)
    for i = 1:nOuter
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i = k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i = k)))
            end

            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        xtLS = [choices[:xtLS=>i=>:LS] for i in 1:nX]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]

        xtCovLog = sum(broadcast(rbfKernelLog, X, X, xtLS))
        logitTcov = processCov(xtCovLog, tScale, tNoise)

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTcov)
        end

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

# Binary Treatment No Confounders No Covariates
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Bool}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)

    n = length(T)

    obs = Gen.choicemap()

    obs[:Y] = Y

    PosteriorSamples = []

    (trace, _) = generate(NoCovNoUBinaryGPSLC, (hyperparams, T), obs)
    for i = 1:nOuter
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples, trace
end

end
