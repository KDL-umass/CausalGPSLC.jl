export Posterior

"""Continuous Treatment Full Model"""
function Posterior(priorparams::PriorParameters, X::Covariates, T::ContinuousTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n, nX = size(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    for k in 1:nX
        obs[:X=>k=>:X] = X[:, k]
    end

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCRealT, (priorparams, n, nU, nX), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k::Int64 = 1:nU
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uyLS", i=k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uxLS", i=k, j=l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xNoise", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xyLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xScale", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        # Algorithm 3 Confounder Update
        uCov = priorparams["SigmaU"] * get_choices(trace)[:uNoise]

        for j = 1:nESInner
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Continuous Treatment No Covariates"""
function Posterior(priorparams::PriorParameters, X::Nothing, T::ContinuousTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)
    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoCovRealT, (priorparams, n, nU, nothing), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uNoise"),))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uyLS", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        # Algorithm 3 Confounder Update
        uCov = priorparams["SigmaU"] * get_choices(trace)[:uNoise]

        for j = 1:nESInner
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""
Continuous Treatment No Confounders

`nESInner` is not used to sample anything via elliptical slice sampling
    It is required here for compatibility with the binary treatment version which uses
    ES to learn hyperparameters for the support of binary variables
    which are not usually supported by Gaussian processes.
"""
function Posterior(priorparams::PriorParameters, X::Covariates, T::ContinuousTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Int64, nESInner::Int64)
    n, nX = size(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoURealT, (priorparams, n, nothing, nX), obs)
    for i = @mock tqdm(1:nOuter)

        for j = 1:nMHInner # Support for loop added after paper
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xyLS", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Continuous Treatment No Confounders No Covariates"""
function Posterior(priorparams::PriorParameters, X::Nothing, T::ContinuousTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Union{Nothing,Int64}, nESInner::Union{Nothing,Int64})
    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:Y] = Y
    obs[:T] = T

    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoUNoCovRealT, (priorparams, n, nothing, nothing), obs)
    for i = @mock tqdm(1:nOuter)
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end


"""Binary Treatment Full Model"""
function Posterior(priorparams::PriorParameters, X::Covariates, T::BinaryTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    # observe invididual T to infer logitT
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    for k in 1:nX
        obs[:X=>k=>:X] = X[:, k]
    end

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCBinaryT, (priorparams, n, nU, nX), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uyLS", i=k)))
                for l = 1:nX
                    (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uxLS", i=k, j=l)))
                end
            end

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xNoise", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xyLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xScale", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        U = [choices[:U=>i=>:U] for i in 1:nU]
        utLS = [choices[:utLS=>i=>:LS] for i in 1:nU]
        xtLS = [choices[:xtLS=>i=>:LS] for i in 1:nX]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]

        utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
        xtCovLog = rbfKernelLog(X, X, xtLS)
        logitTCov = processCov(utCovLog + xtCovLog, tScale, tNoise)

        # Algorithm 3 Confounder Update
        uCov = priorparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTCov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Covariates"""
function Posterior(priorparams::PriorParameters, X::Nothing, T::BinaryTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n = size(T, 1)

    obs = Gen.choicemap()

    obs[:Y] = Y
    # observe invididual T to infer logitT
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoCovBinaryT, (priorparams, n, nU, nothing), obs)
    for i = @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k = 1:nU
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("utLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("uyLS", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        U = [choices[:U=>i=>:U] for i in 1:nU]
        utLS = [choices[:utLS=>i=>:LS] for i in 1:nU]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]
        uNoise = choices[:uNoise]

        utCovLog = sum(broadcast(rbfKernelLog, U, U, utLS))
        logitTCov = processCov(utCovLog, tScale, tNoise)

        # Algorithm 3 Confounder Update
        uCov = priorparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTCov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment with No Confounders"""
function Posterior(priorparams::PriorParameters, X::Covariates, T::BinaryTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    # observe invididual T to infer logitT
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoUBinaryT, (priorparams, n, nothing, nX), obs)
    for i = @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))

            for k = 1:nX
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xtLS", i=k)))
                (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("xyLS", i=k)))
            end

            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tScale")))
            (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))
        end

        choices = get_choices(trace)

        xtLS = [choices[:xtLS=>i=>:LS] for i in 1:nX]
        tScale = choices[:tScale]
        tNoise = choices[:tNoise]

        xtCovLog = rbfKernelLog(X, X, xtLS)
        logitTCov = processCov(xtCovLog, tScale, tNoise)

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logitTCov)
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Confounders No Covariates"""
function Posterior(priorparams::PriorParameters, X::Nothing, T::BinaryTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Union{Nothing,Int64}, nESInner::Union{Nothing,Int64})

    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(CausalGPSLCNoUNoCovBinaryT, (priorparams, n, nothing, nothing), obs)
    for i = @mock tqdm(1:nOuter)
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yNoise")))
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("tyLS")))
        (trace, _) = mh(trace, paramProposal, (priorparams["drift"], getProposalAddress("yScale")))

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end
