export prepareData


"""TODO: Add loadData function to data.jl and have prepareData take DataFrame as input instead"""
function prepareData(csv_path, confounderEps::Float64=1.0e-13, confounderCov::Float64=1.0)
    df = CSV.read(csv_path, DataFrame)
    sort!(df, [:obj])

    # build a list of object size
    # [a, a, a, b, c, c] -> [3, 1, 2]
    counts = Dict()
    for o in df[!, :obj]
        if o in keys(counts)
            counts[o] += 1
        else
            counts[o] = 1
        end
    end
    obj_count = [counts[o] for o in removeAdjacent(df[!, :obj])]
    SigmaU = generateSigmaU(obj_count, confounderEps, confounderCov)

    # prepare inputs
    T = Array(df[!, :T])
    Y = Array(df[!, :Y])
    n = size(T, 1)

    cols = names(df)
    cols = deleteat!(cols, cols .== "T")
    cols = deleteat!(cols, cols .== "Y")
    cols = deleteat!(cols, cols .== "obj")
    if length(cols) == 0
        X = nothing
    else
        X_ = df[!, cols]
        nX = size(X_)[2]
        X = zeros(n, nX)
        for i in 1:nX
            X[:, i] = Vector(X_[!, i])
        end
    end

    return X, T, Y, SigmaU
end