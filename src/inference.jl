export Posterior

"""Continuous Treatment Full Model"""
function Posterior(hyperparams::Dict, X::Covariates, T::ContinuousTreatment, Y::Outcome,
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

    (trace, _) = generate(GPSLCRealT, (hyperparams, nU, nX), obs)
    for i in @mock tqdm(1:nOuter)
        for j = 1:nMHInner
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("uNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("yNoise")))
            (trace, _) = mh(trace, paramProposal, (0.5, getProposalAddress("tyLS")))

            for k::Int64 = 1:nU
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

        # Algorithm 3 Confounder Update
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

"""Continuous Treatment No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::ContinuousTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)
    n = size(T, 1)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    # Algorithm 2 HyperParameter Update
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

        # Algorithm 3 Confounder Update
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

"""
Continuous Treatment No Confounders

`nESInner` is not used to sample anything via elliptical slice sampling
    It is required here for compatibility with the binary treatment version which uses
    ES to learn hyperparameters for the support of binary variables
    which are not usually supported by Gaussian processes.
"""
function Posterior(hyperparams::Dict, X::Covariates, T::ContinuousTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Int64, nESInner::Int64)
    n, nX = size(X)

    obs = Gen.choicemap()
    obs[:T] = T
    obs[:Y] = Y

    # Algorithm 2 HyperParameter Update
    posteriorSamples = []

    (trace, _) = generate(GPSLCNoURealT, (hyperparams, X), obs)
    for i = @mock tqdm(1:nOuter)

        for j = 1:nMHInner # Support for loop added after paper
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

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Continuous Treatment No Confounders No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::ContinuousTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Nothing, nESInner::Nothing)
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
function Posterior(hyperparams::Dict, X::Covariates, T::BinaryTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    for k in 1:nX
        obs[:X=>k=>:X] = X[:, k]
    end

    # Algorithm 2 HyperParameter Update
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
        xtCovLog = rbfKernelLog(X, X, xtLS)
        logittCov = processCov(utCovLog + xtCovLog, tScale, tNoise)

        # Algorithm 3 Confounder Update
        uCov = hyperparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logittCov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::BinaryTreatment, Y::Outcome,
    nU::Int64, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n = size(T, 1)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    # Algorithm 2 HyperParameter Update
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
        logittCov = processCov(utCovLog, tScale, tNoise)

        # Algorithm 3 Confounder Update
        uCov = hyperparams["SigmaU"] * uNoise

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logittCov)
            for k = 1:nU
                trace = elliptical_slice(trace, :U => k => :U, zeros(n), uCov)
            end
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment with No Confounders"""
function Posterior(hyperparams::Dict, X::Covariates, T::BinaryTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Int64, nESInner::Int64)

    n, nX = size(X)

    obs = Gen.choicemap()

    obs[:Y] = Y
    for i in 1:n
        obs[:T=>i=>:T] = T[i]
    end

    # Algorithm 2 HyperParameter Update
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

        xtCovLog = rbfKernelLog(X, X, xtLS)
        logittCov = processCov(xtCovLog, tScale, tNoise)

        for j = 1:nESInner
            trace = elliptical_slice(trace, :logitT, zeros(n), logittCov)
        end

        push!(posteriorSamples, get_choices(trace))
    end
    posteriorSamples, trace
end

"""Binary Treatment No Confounders No Covariates"""
function Posterior(hyperparams::Dict, X::Nothing, T::BinaryTreatment, Y::Outcome,
    nU::Nothing, nOuter::Int64, nMHInner::Nothing, nESInner::Nothing)

    n = size(T, 1)

    obs = Gen.choicemap()

    obs[:Y] = Y

    # Algorithm 2 HyperParameter Update
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
