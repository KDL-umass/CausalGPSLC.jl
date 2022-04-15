export Posterior

"""Full Model"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Float64}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n, nX = size(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    for i in 1:nX
        obs[:X=>i=>:X] = X[i, :]
    end

    posteriorSamples = []

    (trace, _) = generate(GPSLCRealT, (hyperparams, nU, nX), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k::Int = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i=k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uxLS", i=k, j=l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xNoise", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xScale", i=k)))
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Float64}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)
    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoCovRealT, (hyperparams, nU), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise"),))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i=k)))
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""No latent confounder, continuous treatment"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Float64}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)
    n, nX = size(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoURealT, (hyperparams, X), obs)
    for i = @mock tqdm(1:nOuter)
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

        for k = 1:nX
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i=k)))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i=k)))
        end

        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tScale")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""No latent confounders or covariates, continuous treatment"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Float64}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)
    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:Y] = Y

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoUNoCovRealT, (hyperparams, T), obs)
    for i = @mock tqdm(1:nOuter)
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end


"""Binary Treatment Full Model"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Bool}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    for i in 1:nX
        obs[:X=>i=>:X] = X[i, :]
    end

    posteriorSamples = []

    (trace, _) = generate(GPSLCBinaryT, (hyperparams, nU, nX), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i=k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uxLS", i=k, j=l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xNoise", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xScale", i=k)))
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Bool}, Y::Array{Float64},
    nU::Int, nOuter::Int, nMHInner::Int, nESInner::Int)

    n = size(T, 1)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoCovBinaryT, (hyperparams, nU), obs)
    for i = @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uyLS", i=k)))
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment with No Confounders"""
function Posterior(hyperparams::Dict, X::Array{Array{Float64,1}}, T::Array{Bool}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Int, nESInner::Int)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoUBinaryT, (hyperparams, X), obs)
    for i = @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("xyLS", i=k)))
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Confounders No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::Array{Bool}, Y::Array{Float64},
    nU::Nothing, nOuter::Int, nMHInner::Nothing, nESInner::Nothing)

    n = size(T, 1)

    obs = Gen.choicemap()

    obs[:Y] = Y

    posteriorSamples = []

    (trace, _) = generate(GPSLCNoUNoCovBinaryT, (hyperparams, T), obs)
    for i = @mock tqdm(1:nOuter)
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yScale")))

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end
