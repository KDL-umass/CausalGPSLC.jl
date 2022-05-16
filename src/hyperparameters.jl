export getPriorParameters

"""
    PriorParameters

Defaults are those used in original paper, listed here for modification

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
    )
end

"""
    HyperParameters

Returns default values for hyperparametrs

- `nU = 1`
- `nOuter = 20`
- `nInner = 5`
- `nESInner = 5`
- `nBurnIn = 5`
- `stepSize = 1`
"""
function getHyperParameters()::HyperParameters
    nU = 1
    nOuter = 24
    nInner = 10
    nESInner = 5
    nBurnIn = 10
    stepSize = 1
    predictionCovarianceNoise = 1e-10
    HyperParameters(
        nU,
        nOuter,
        nInner,
        nESInner,
        nBurnIn,
        stepSize,
        predictionCovarianceNoise
    )
end