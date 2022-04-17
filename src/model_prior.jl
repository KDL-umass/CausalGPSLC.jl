export sampleNoiseFromPriorU, sampleNoiseFromPriorX, sampleNoiseFromPriorT, sampleNoiseFromPriorY, lengthscaleFromPriorT, lengthscaleFromPriorU, lengthscaleFromPriorX, scaleFromPriorX, scaleFromPriorT, scaleFromPriorY

"""Gen function to generate lengthscale parameter for GP"""
@gen function generateLS(shape, scale)::Float64
    @trace(inv_gamma(shape, scale), :LS)
end

"""Gen function to generate scale parameter for GP"""
@gen function generateScale(shape, scale)::Float64
    @trace(inv_gamma(shape, scale), :Scale)
end

"""Gen function to generate noise from inv_gamma"""
@gen function generateNoise(shape, scale)::Float64
    @trace(inv_gamma(shape, scale), :Noise)
end

"""Gen function to generate binary treatment (T)"""
@gen function generateBinaryT(logitT)::Bool
    @trace(bernoulli(expit(logitT)), :T)
end

"""Gen function to generate latent confounders (U) from mvnormal distribution"""
@gen function generateU(Ucov::Array{Float64}, n::Int)::Vector{Float64}
    @assert size(Ucov) == (n, n) "Ucov is not NxN"
    @trace(mvnormal(zeros(n), Ucov), :U)
end

"""Gen function to generate covariates (n,X_k) from mvnormal distribution"""
@gen function generateX(Xcov_k::Matrix{Float64}, n::Int)
    @trace(mvnormal(zeros(n), Xcov_k), :X)
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

Sample noise for prior from confounders (U)
"""
@gen function sampleNoiseFromPriorU(hyperparams::HyperParameters)
    uNoise = @trace(inv_gamma(hyperparams["uNoiseShape"], hyperparams["uNoiseScale"]), :uNoise)
    return uNoise
end


"""Sample noise from prior for covariates (X)"""
@gen function sampleNoiseFromPriorX(hyperparams::HyperParameters, nX::Int64)
    xNoise = @trace(MappedGenerateNoise(fill(hyperparams["xNoiseShape"], nX),
            fill(hyperparams["xNoiseScale"], nX)), :xNoise)

    return xNoise
end

"""Sample noise from prior for treatment (T)"""
@gen function sampleNoiseFromPriorT(hyperparams::HyperParameters)
    tNoise = @trace(inv_gamma(hyperparams["tNoiseShape"], hyperparams["tNoiseScale"]), :tNoise)

    return tNoise
end

"""Sample noise from prior for outcome (Y)"""
@gen function sampleNoiseFromPriorY(hyperparams::HyperParameters)
    yNoise = @trace(inv_gamma(hyperparams["yNoiseShape"], hyperparams["yNoiseScale"]), :yNoise)
    return yNoise
end


"""Treatment to outcome lengthscale"""
@gen function lengthscaleFromPriorT(hyperparams::HyperParameters)
    tyLS = @trace(inv_gamma(hyperparams["tyLSShape"], hyperparams["tyLSScale"]), :tyLS)
    return tyLS
end

"""Latent confounders to treatment and outcome lengthscale"""
@gen function lengthscaleFromPriorU(hyperparams::HyperParameters, nU::Int64)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU),
            fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU),
            fill(hyperparams["uyLSScale"], nU)), :uyLS)
    return utLS, uyLS
end

"""Latent confounders to treatment and outcome lengthscale when nX is known"""
@gen function lengthscaleFromPriorUX(hyperparams::HyperParameters, nU::Int64, nX::Int64)
    uxLS = @trace(MappedMappedGenerateLS(fill(fill(hyperparams["uxLSShape"], nX), nU), fill(fill(hyperparams["uxLSScale"], nX), nU)), :uxLS)
    uxLS = toMatrix(uxLS, nX, nU)
    utLS = @trace(MappedGenerateLS(fill(hyperparams["utLSShape"], nU), fill(hyperparams["utLSScale"], nU)), :utLS)
    uyLS = @trace(MappedGenerateLS(fill(hyperparams["uyLSShape"], nU), fill(hyperparams["uyLSScale"], nU)), :uyLS)
    return uxLS, utLS, uyLS
end

"""Covariates to treatment and outcome lengthscale"""
@gen function lengthscaleFromPriorX(hyperparams::HyperParameters, nX::Int64)
    xtLS = @trace(MappedGenerateLS(fill(hyperparams["xtLSShape"], nX),
            fill(hyperparams["xtLSScale"], nX)), :xtLS)
    xyLS = @trace(MappedGenerateLS(fill(hyperparams["xyLSShape"], nX),
            fill(hyperparams["xyLSScale"], nX)), :xyLS)
    @assert size(xtLS, 1) == nX "x lengthscale not correct length"
    @assert size(xyLS, 1) == nX "x lengthscale not correct length"
    return xtLS, xyLS
end



"""
Generate kernel scales from prior for covariates (X)
"""
@gen function scaleFromPriorX(hyperparams::HyperParameters, nX::Int64)
    xScale = @trace(MappedGenerateScale(fill(hyperparams["xScaleShape"], nX), fill(hyperparams["xScaleScale"], nX)), :xScale)
    return xScale
end

"""
Sample kernel scale from prior for treatment (T)
"""
@gen function scaleFromPriorT(hyperparams::HyperParameters)
    tScale = @trace(inv_gamma(hyperparams["tScaleShape"], hyperparams["tScaleScale"]), :tScale)
    return tScale
end

"""
Sample kernel scale from prior for outcome (Y)
"""
@gen function scaleFromPriorY(hyperparams::HyperParameters)
    yScale = @trace(inv_gamma(hyperparams["yScaleShape"], hyperparams["yScaleScale"]), :yScale)
    return yScale
end