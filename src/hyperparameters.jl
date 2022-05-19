export getPriorParameters, getHyperParameters

"""
    getPriorParameters()

These are standard values for scale and shape of Inverse Gamma priors over kernel parameters, confounder structure covariance noise, and confounder Gaussian prior covariance. 

- `uNoiseShape::Float64=4.0`: shape of the InvGamma prior over the noise of U
- `uNoiseScale::Float64=4.0`: scale of the InvGamma prior over the noise of U
- `xNoiseShape::Float64=4.0`: shape of the InvGamma prior over the noise of X
- `xNoiseScale::Float64=4.0`: scale of the InvGamma prior over the noise of X
- `tNoiseShape::Float64=4.0`: shape of the InvGamma prior over the noise of T
- `tNoiseScale::Float64=4.0`: scale of the InvGamma prior over the noise of T
- `yNoiseShape::Float64=4.0`: shape of the InvGamma prior over the noise of Y
- `yNoiseScale::Float64=4.0`: scale of the InvGamma prior over the noise of Y
- `xScaleShape::Float64=4.0`: shape of the InvGamma prior over kernel scale of X
- `xScaleScale::Float64=4.0`: scale of the InvGamma prior over kernel scale of X
- `tScaleShape::Float64=4.0`: shape of the InvGamma prior over kernel scale of T
- `tScaleScale::Float64=4.0`: scale of the InvGamma prior over kernel scale of T
- `yScaleShape::Float64=4.0`: shape of the InvGamma prior over kernel scale of Y
- `yScaleScale::Float64=4.0`: scale of the InvGamma prior over kernel scale of Y
- `uxLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of U and X
- `uxLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of U and X
- `utLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of U and T
- `utLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of U and T
- `xtLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of X and T
- `xtLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of X and T
- `uyLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of U and Y
- `uyLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of U and Y
- `xyLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of X and Y
- `xyLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of X and Y
- `tyLSShape::Float64=4.0`: shape of the InvGamma prior over kernel lengthscale of T and Y
- `tyLSScale::Float64=4.0`: scale of the InvGamma prior over kernel lengthscale of T and Y
- `sigmaUNoise::Float64=1.0e-13`: noise added to matrix to make covariance stable and invertible
- `sigmaUCov::Float64=1.0`: assumed covariance over structured confounders
- `drift::Float64=0.5`: as in the paper, Metropolis Hastings Gaussian Drift
"""
function getPriorParameters()::PriorParameters
    Dict{String,Any}(
        "uNoiseShape" => 4.0,
        "uNoiseScale" => 4.0,
        "xNoiseShape" => 4.0,
        "xNoiseScale" => 4.0,
        "tNoiseShape" => 4.0,
        "tNoiseScale" => 4.0,
        "yNoiseShape" => 4.0,
        "yNoiseScale" => 4.0,
        "xScaleShape" => 4.0,
        "xScaleScale" => 4.0,
        "tScaleShape" => 4.0,
        "tScaleScale" => 4.0,
        "yScaleShape" => 4.0,
        "yScaleScale" => 4.0,
        "uxLSShape" => 4.0,
        "uxLSScale" => 4.0,
        "utLSShape" => 4.0,
        "utLSScale" => 4.0,
        "xtLSShape" => 4.0,
        "xtLSScale" => 4.0,
        "uyLSShape" => 4.0,
        "uyLSScale" => 4.0,
        "xyLSShape" => 4.0,
        "xyLSScale" => 4.0,
        "tyLSShape" => 4.0,
        "tyLSScale" => 4.0,
        "sigmaUNoise" => 1.0e-13,
        "sigmaUCov" => 1.0,
        "drift" => 0.5,
    )
end

"""
    getHyperParameters()

Returns default values for hyperparameters

- `nU = 1`: Number of latent confounding variables assumed to be influencing all the instances that belong to one object. Inference will be performed over these values.
- `nOuter = 20`: Number of posterior samples to draw.
- `nMHInner = 5`: Number of internal Metropolis-Hastings updates to make per posterior sample.
- `nESInner = 5`: Number of elliptical-slice sampling updates to make per posterior for latent confounders and binary treatment.
- `nBurnIn = 5`: Number of posterior samples to discard when making predictions and estimates.
- `stepSize = 1`: How frequently to use posterior samples (1 being every one after burnIn, higher being every `stepSize`_th_).
- `predictionCovarianceNoise=1e-10`: Predicting with Gaussian processes requires use of covariance matrices that are Symmetric Positive Definite, and this covariance noise on the diagonal ensures these operations can be performed in a stable and consistent way.
"""
function getHyperParameters()::HyperParameters
    nU = 1
    nOuter = 24
    nMHInner = 10
    nESInner = 5
    nBurnIn = 10
    stepSize = 1
    predictionCovarianceNoise = 1e-10
    HyperParameters(
        nU,
        nOuter,
        nMHInner,
        nESInner,
        nBurnIn,
        stepSize,
        predictionCovarianceNoise
    )
end