export prepareData

function prepareData(csv_path)
    df = CSV.read(csv_path, DataFrame)

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
    SigmaU = generateSigmaU(obj_count)

    # prepare inputs
    T = Array(df[!, :T])
    Y = Array(df[!, :Y])

    cols = names(df)
    cols = deleteat!(cols, cols .== "T")
    cols = deleteat!(cols, cols .== "Y")
    cols = deleteat!(cols, cols .== "obj")
    if length(cols) == 0
        X = nothing
    else
        X_ = df[!, cols]
        nX = size(X_)[2]
        X = [Array(X_[!, i]) for i in 1:nX]
    end

    X, T, Y, SigmaU
end