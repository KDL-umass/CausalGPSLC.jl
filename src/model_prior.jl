export sampleNoiseFromPrior, lengthscaleFromPrior

"""Gen function to generate lengthscale parameter for GP"""
@gen function generateLS(shape, scale)
    @trace(inv_gamma(shape, scale), :LS)
end

"""Gen function to generate scale parameter for GP"""
@gen function generateScale(shape, scale)
    @trace(inv_gamma(shape, scale), :Scale)
end

"""Gen function to generate noise from inv_gamma"""
@gen function generateNoise(shape, scale)
    @trace(inv_gamma(shape, scale), :Noise)
end

"""Gen function to generate binary treatment (T)"""
@gen function generateBinaryT(logitT)
    @trace(bernoulli(expit(logitT)), :T)
end

"""Gen function to generate latent confounders (U) from mvnormal distribution"""
@gen function generateU(Ucov::Array{Float64}, n::Int)
    @trace(mvnormal(fill(0, n), Ucov), :U)
end

"""Gen function to generate covariates (X) from mvnormal distribution"""
@gen function generateX(Xcov::Array{Float64}, n::Int)
    @trace(mvnormal(fill(0, n), Xcov), :X)
end

export MappedGenerateLS, MappedMappedGenerateLS, MappedGenerateScale, MappedGenerateBinaryT, MappedGenerateNoise, MappedGenerateU, MappedGenerateX

MappedGenerateLS = Map(generateLS)
MappedMappedGenerateLS = Map(MappedGenerateLS)
MappedGenerateScale = Map(generateScale)
MappedGenerateBinaryT = Map(generateBinaryT)
MappedGenerateNoise = Map(generateNoise)
MappedGenerateU = Map(generateU)
MappedGenerateX = Map(generateX)


load_generated_functions()

"""
Generate noise terms from noise prior
"""
@gen function sampleNoiseFromPrior(hyperparams::HyperParameters, nX)
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    xNoise = @trace(MappedGenerateNoise(fill(hyperparams["xNoiseShape"], nX),
            fill(hyperparams["xNoiseScale"], nX)), :xNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)

    return uNoise, xNoise, tNoise, yNoise
end

@gen function sampleNoiseFromPrior(hyperparams::HyperParameters)
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    return uNoise, tNoise, yNoise
end

"""Treatment to outcome lengthscale"""
@gen function sampleTYLengthscale(hyperparams::HyperParameters)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    return tyLS
end

"""Latent confounders to treatment and outcome lengthscale"""
@gen function sampleUtUyLengthscale(hyperparams::HyperParameters)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    return utLS, uyLS
end

"""Covariates to treatment and outcome lengthscale"""
@gen function sampleXtXyLengthscale(hyperparams::HyperParameters, nX)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX),
            fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX),
            fill(hyperparams["xyLSScale"], nX)), :xyLS)
    return xtLS, xyLS
end


"""
Generate kernel lengthscales from prior with U and X
"""
@gen function lengthscaleFromPriorUX(hyperparams::HyperParameters, nU::Int64, nX::Int64)
    utLS, uyLS = @trace(sampleUtUyLengthscale(hyperparams))

    uxLS = @trace(MappedMappedGenerateLS(fill(fill(hyperparams["uxLSShape"], nX), nU),
            fill(fill(hyperparams["uxLSScale"], nX), nU)), :uxLS)

    tyLS = @trace(sampleTYLengthscale(hyperparams))
    xtLS, xyLS = sampleXtXyLengthscale(hyperparams, nX)

    return utLS, uyLS, uxLS, tyLS, xtLS, xyLS
end

"""
Generate kernel lengthscales from prior without covariates
"""
@gen function lengthscaleFromPriorU(hyperparams::HyperParameters, nU::Int64, nX::Nothing)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    return utLS, uyLS, tyLS
end

"""
Generate kernel lengthscales from prior without latent confounders
"""
@gen function lengthscaleFromPriorX(hyperparams::HyperParameters, nU::Nothing, nX::Int64)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    xtLS, xyLS = @trace(sampleXtXyLengthscale(hyperparams, nX))
    return tyLS, xtLS, xyLS
end

"""
Generate kernel lengthscales from prior without latent confounders or covariates
"""
@gen function lengthscaleFromPrior(hyperparams::HyperParameters, nU::Nothing, nX::Nothing)
    tyLS = @trace(sampleTYLengthscale(hyperparams))
    return tyLS
end