export sampleNoiseFromPriorU, sampleNoiseFromPriorX, sampleNoiseFromPriorT,
    sampleNoiseFromPriorY, lengthscaleFromPriorT, lengthscaleFromPriorU,
    lengthscaleFromPriorX, scaleFromPriorX, scaleFromPriorT, scaleFromPriorY,
    generateXfromPrior, generateRealTfromPrior, generateBinaryTfromPrior

"""Gen function to generate lengthscale parameter for GP"""
@gen function generateLS(shape::Float64, scale::Float64)::Float64
    @trace(inv_gamma(shape, scale), :LS)
end

"""Gen function to generate scale parameter for GP"""
@gen function generateScale(shape::Float64, scale::Float64)::Float64
    @trace(inv_gamma(shape, scale), :Scale)
end

"""Gen function to generate noise from inv_gamma"""
@gen function generateNoise(shape::Float64, scale::Float64)::Float64
    @trace(inv_gamma(shape, scale), :Noise)
end

"""Gen function to generate binary treatment (T) from real value"""
@gen function generateBinaryT(logitT::Float64)::Bool
    @trace(bernoulli(expit(logitT)), :T)
end

"""Gen function to generate latent confounders (U) from mvnormal distribution"""
@gen function generateU(uCov::Array{Float64}, n::Int64)::Vector{Float64}
    @assert size(uCov) == (n, n) "uCov is not NxN"
    @trace(mvnormal(zeros(n), uCov), :U)
end

"""Gen function to generate covariates (n,X_k) from mvnormal distribution"""
@gen function generateX(XCov_k::Matrix{Float64}, n::Int64)::Vector{Float64}
    @assert size(XCov_k) == (n, n) "XCov_k is not NxN"
    @trace(mvnormal(zeros(n), XCov_k), :X)
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
@gen function sampleNoiseFromPriorU(priorparams::PriorParameters)::Float64
    uNoise = @trace(inv_gamma(priorparams["uNoiseShape"], priorparams["uNoiseScale"]), :uNoise)
    return uNoise
end


"""Sample noise from prior for covariates (X)"""
@gen function sampleNoiseFromPriorX(priorparams::PriorParameters, nX::Int64)::Vector{Float64}
    xNoise = @trace(MappedGenerateNoise(
            fill(priorparams["xNoiseShape"], nX),
            fill(priorparams["xNoiseScale"], nX)
        ), :xNoise)
    return xNoise
end

"""Sample noise from prior for treatment (T)"""
@gen function sampleNoiseFromPriorT(priorparams::PriorParameters)::Float64
    tNoise = @trace(inv_gamma(priorparams["tNoiseShape"], priorparams["tNoiseScale"]), :tNoise)

    return tNoise
end

"""Sample noise from prior for outcome (Y)"""
@gen function sampleNoiseFromPriorY(priorparams::PriorParameters)::Float64
    yNoise = @trace(inv_gamma(priorparams["yNoiseShape"], priorparams["yNoiseScale"]), :yNoise)
    return yNoise
end


"""Treatment to outcome lengthscale"""
@gen function lengthscaleFromPriorT(priorparams::PriorParameters)::Float64
    tyLS = @trace(inv_gamma(priorparams["tyLSShape"], priorparams["tyLSScale"]), :tyLS)
    return tyLS
end

"""Latent confounders to treatment and outcome lengthscale"""
@gen function lengthscaleFromPriorU(priorparams::PriorParameters, nU::Int64)::Tuple{Vector{Float64},Vector{Float64}}
    utLS = @trace(MappedGenerateLS(
            fill(priorparams["utLSShape"], nU),
            fill(priorparams["utLSScale"], nU)
        ), :utLS)
    uyLS = @trace(MappedGenerateLS(
            fill(priorparams["uyLSShape"], nU),
            fill(priorparams["uyLSScale"], nU)
        ), :uyLS)
    return utLS, uyLS
end

"""Latent confounders to treatment and outcome lengthscale when nX is known"""
@gen function lengthscaleFromPriorUX(priorparams::PriorParameters, nU::Int64, nX::Int64)::Tuple{Matrix{Float64},Vector{Float64},Vector{Float64}}
    uxLS = @trace(MappedMappedGenerateLS(
            fill(fill(priorparams["uxLSShape"], nX), nU),
            fill(fill(priorparams["uxLSScale"], nX), nU)
        ), :uxLS)
    uxLS = toMatrix(uxLS, nX, nU)
    utLS = @trace(MappedGenerateLS(
            fill(priorparams["utLSShape"], nU),
            fill(priorparams["utLSScale"], nU)
        ), :utLS)
    uyLS = @trace(MappedGenerateLS(
            fill(priorparams["uyLSShape"], nU),
            fill(priorparams["uyLSScale"], nU)
        ), :uyLS)
    return uxLS, utLS, uyLS
end

"""Covariates to treatment and outcome lengthscale"""
@gen function lengthscaleFromPriorX(priorparams::PriorParameters, nX::Int64)::Tuple{Vector{Float64},Vector{Float64}}
    xtLS = @trace(MappedGenerateLS(
            fill(priorparams["xtLSShape"], nX),
            fill(priorparams["xtLSScale"], nX)
        ), :xtLS)
    xyLS = @trace(MappedGenerateLS(
            fill(priorparams["xyLSShape"], nX),
            fill(priorparams["xyLSScale"], nX)
        ), :xyLS)
    @assert size(xtLS, 1) == nX "x lengthscale not correct length"
    @assert size(xyLS, 1) == nX "x lengthscale not correct length"
    return xtLS, xyLS
end



"""
Generate kernel scales from prior for covariates (X)
"""
@gen function scaleFromPriorX(priorparams::PriorParameters, nX::Int64)::Vector{Float64}
    xScale = @trace(MappedGenerateScale(
            fill(priorparams["xScaleShape"], nX),
            fill(priorparams["xScaleScale"], nX)
        ), :xScale)
    return xScale
end

"""
Sample kernel scale from prior for treatment (T)
"""
@gen function scaleFromPriorT(priorparams::PriorParameters)::Float64
    tScale = @trace(inv_gamma(
            priorparams["tScaleShape"],
            priorparams["tScaleScale"]
        ), :tScale)
    return tScale
end

"""
Sample kernel scale from prior for outcome (Y)
"""
@gen function scaleFromPriorY(priorparams::PriorParameters)::Float64
    yScale = @trace(inv_gamma(
            priorparams["yScaleShape"],
            priorparams["yScaleScale"]
        ), :yScale)
    return yScale
end

"""
Sample covariates from prior (X)
"""
@gen function generateXfromPrior(priorparams::PriorParameters, n::Int64, nX::Int64)::Covariates
    X = zeros(n, nX)
    for k in 1:nX
        X[:, k] = @trace(mvnormal(zeros(n), LinearAlgebra.I(n)), :X => k => :X)
    end
    return X
end


"""
Sample continuous treatment from prior (T)
"""
@gen function generateRealTfromPrior(priorparams::PriorParameters, n::Int64)::ContinuousTreatment
    @trace(mvnormal(zeros(n), LinearAlgebra.I(n)), :T)
end


"""
Sample binary treatment from prior (T)
"""
@gen function generateBinaryTfromPrior(priorparams::PriorParameters, n::Int64)::BinaryTreatment
    logitTCov = LinearAlgebra.I(n)
    logitT = @trace(mvnormal(zeros(n), logitTCov), :logitT)
    T = @trace(MappedGenerateBinaryT(logitT), :T)
    return T
end