module MultilevelModel

# libraries
using Gen
using ProgressBars
using LinearAlgebra
using Statistics

export posteriorLinearMLM, posteriorLinearMLMoffset

# parameter with normal prior
@gen function generateLS(mean, scale)
    LS = @trace(normal(mean, scale), :LS)
    return LS
end

# parameter with inverse gamma prior
@gen function generateNoise(shape, scale)
    Noise = @trace(inv_gamma(shape, scale), :noise)
    return Noise
end

# method for random walk
@gen function thetaProposal1(trace, var::Float64)
    mu = trace[:theta]
    @trace(normal(mu, var), :theta)
end

# method for random walk
@gen function thetaProposal2(trace, i::Int, var::Float64)
    mu = trace[:theta => i => :LS]
    @trace(normal(mu, var), :theta => i => :LS)
end

@gen function alphaProposal(trace, i::Int, var::Float64)
    mu = trace[:alpha => i => :LS]
    @trace(normal(mu, var), :alpha => i => :LS)
end

@gen function betaProposal(trace, i::Int, var::Float64)
    mu = trace[:beta => i => :LS]
    @trace(normal(mu, var), :beta => i => :LS)
end

@gen function NoiseProposal(trace, var::Float64)
    cur = trace[:noise]

    Shape = (cur * cur / var) + 2
    Scale = cur * (Shape - 1)

    @trace(inv_gamma(Shape, Scale), :noise)
end

MappedGenerateLS = Map(generateLS)
load_generated_functions()


# Multilevel with object-wide offsets
@gen function LinearMLMOffsetwithX(xs::Vector{Float64}, ts::Vector{Float64}, obj_label)
    n, nX = size(xs)
    nObj = length(Set(obj_label))
    beta =  @trace(MappedGenerateLS(fill(0.0, nX), fill(1.0, nX)), :beta) #xyLS
    theta = @trace(normal(0,1), :theta) #tyLS
    alpha = @trace(MappedGenerateLS(fill(0.0, nObj), fill(10.0, nObj)), :alpha) #alpha
    sigma = @trace(inv_gamma(4.0, 4.0), :noise) #tyLS
    for i in 1:n
        obj = Int(obj_label[i])
        t = ts[i]
        x = xs[i, :]
        @trace(normal( sum(beta.*x) +theta * t + alpha[obj], sigma), "y-$i")
    end
end


# Multilevel with object-wide offsets. No covariates
@gen function LinearMLMOffset(ts::Vector{Float64}, obj_label)
    nObj = length(Set(obj_label))
    n = length(ts)
    theta = @trace(normal(0,1), :theta) #tyLS
    alpha = @trace(MappedGenerateLS(fill(0.0, nObj), fill(10.0, nObj)), :alpha) #alpha
    sigma = @trace(inv_gamma(4.0, 4.0), :noise) #tyLS
    for i in 1:n
        obj = Int(obj_label[i])
        t = ts[i]
        @trace(normal( theta * t + alpha[obj], sigma), "y-$i")
    end
end


# Multilevel with object-wide offsets and treatment effects
@gen function LinearMLMwithX(xs::Vector{Float64}, ts::Vector{Float64}, obj_label)
    n, nX = size(xs)
    nObj = length(Set(obj_label))
    beta =  @trace(MappedGenerateLS(fill(0.0, nX), fill(1.0, nX)), :beta) #xyLS
    theta = @trace(MappedGenerateLS(fill(0.0, nObj), fill(1.0, nObj)),  :theta) #alpha
    alpha = @trace(MappedGenerateLS(fill(0.0, nObj), fill(10.0, nObj)), :alpha) #alpha
    sigma = @trace(inv_gamma(4.0, 4.0), :noise) #tyLS
    for i in 1:n
        obj = Int(obj_label[i])
        t = ts[i]
        x = xs[i, :]
        @trace(normal( sum(beta.*x) +theta[obj] * t + alpha[obj], sigma), "y-$i")
    end
end


# Multilevel with object-wide offsets and treatment effects. No covariates
@gen function LinearMLM(ts::Vector{Float64}, obj_label)
    n = length(ts)
    nObj = length(Set(obj_label))
    theta = @trace(MappedGenerateLS(fill(0.0, nObj), fill(1.0, nObj)),  :theta) #alpha
    alpha = @trace(MappedGenerateLS(fill(0.0, nObj), fill(10.0, nObj)), :alpha) #alpha
    sigma = @trace(inv_gamma(4.0, 4.0), :noise) #tyLS
    for i in 1:n
        obj = Int(obj_label[i])
        t = ts[i]
        @trace(normal(theta[obj] * t + alpha[obj], sigma), "y-$i")
    end
end

# inference code here
function posteriorLinearMLM(n_samples::Int, T::Vector{Float64}, X::Array{Float64, 2}, Y::Vector{Float64}, obj_label::Vector{Int})
    constraints = Gen.choicemap()
    for (i, y) in enumerate(Y)
        constraints["y-$i"] = y
    end
    PosteriorSamples = []
    nObj = length(Set(obj_label))
    (trace, _) = generate(LinearMLMwithX, (X, T, obj_label), constraints)
    for iter=tqdm(1:n_samples)
        (trace, _) = mh(trace, NoiseProposal, (0.5, ))
        for k in 1:nObj
            (trace, _) = mh(trace, thetaProposal2, (k, 0.5))
            (trace, _) = mh(trace, alphaProposal, (k, 0.5))
        end
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples
end


function posteriorLinearMLM(n_samples::Int, T::Vector{Float64}, Y::Vector{Float64}, obj_label::Vector{Int})
    constraints = Gen.choicemap()
    for (i, y) in enumerate(Y)
        constraints["y-$i"] = y
    end
    PosteriorSamples = []
    nObj = length(Set(obj_label))
    (trace, _) = generate(LinearMLM, (T, obj_label), constraints)
    for iter=tqdm(1:n_samples)

        (trace, _) = mh(trace, NoiseProposal, (0.5, ))
        for k in 1:nObj
            (trace, _) = mh(trace, thetaProposal2, (k, 0.5))
            (trace, _) = mh(trace, alphaProposal, (k, 0.5))
        end
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples
end


function posteriorLinearMLMoffset(n_samples::Int, T::Vector{Float64}, X::Array{Float64, 2}, Y::Vector{Float64}, obj_label::Vector{Int})
    constraints = Gen.choicemap()
    for (i, y) in enumerate(Y)
        constraints["y-$i"] = y
    end
    nObj = length(Set(obj_label))
    PosteriorSamples = []
    (trace, _) = generate(LinearMLMOffsetwithX, (X, T, obj_label), constraints)
    for iter=tqdm(1:n_samples)
        (trace, _) = mh(trace, thetaProposal1, (0.5, ))
        (trace, _) = mh(trace, NoiseProposal, (0.5, ))
        for k in 1:nObj
            (trace, _) = mh(trace, alphaProposal, (k, 0.5))
        end
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples
end


function posteriorLinearMLMoffset(n_samples::Int, T::Vector{Float64}, Y::Vector{Float64}, obj_label::Vector{Int})
    constraints = Gen.choicemap()
    for (i, y) in enumerate(Y)
        constraints["y-$i"] = y
    end
    PosteriorSamples = []
    nObj = length(Set(obj_label))
    (trace, _) = generate(LinearMLMOffset, (T, obj_label), constraints)
    for iter=tqdm(1:n_samples)
        (trace, _) = mh(trace, thetaProposal1, (0.5, ))
        (trace, _) = mh(trace, NoiseProposal, (0.5, ))
        for k in 1:nObj
            (trace, _) = mh(trace, alphaProposal, (k, 0.5))
        end
        push!(PosteriorSamples, get_choices(trace))
    end
    PosteriorSamples
end

end