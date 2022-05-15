export prepareData

"""Returns dataframe of csv at path."""
function loadData(csv_path)
    CSV.read(csv_path, DataFrame)
end

"""
    Prepare Data

Creates the latent confounding structure from the object labels in the data.

Parses matrices for the observed covariates, treatments, and outcomes.

Returns: `X, T, Y, SigmaU`
"""
function prepareData(df::Union{DataFrame,String}, confounderEps::Float64=1.0e-13, confounderCov::Float64=1.0)
    if typeof(df) == String
        df::DataFrame = loadData(df)
    end
    if "obj" in names(df)
        DataFrames.sort!(df, :obj)

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
        obj = Array(df[!, :obj])
        obj_count = [counts[o] for o in removeAdjacent(df[!, :obj])]
        SigmaU = generateSigmaU(obj_count, confounderEps, confounderCov)
    else
        println("No object labels to assign latent confounders to 
                (column must be titled `obj`)")
        obj = nothing
        println("Assuming no latent confounding")
    end

    # prepare inputs
    T = Array(df[!, :T])
    Y = Array(df[!, :Y])
    n = size(T, 1)

    cols = names(df)
    cols = deleteat!(cols, cols .== "T")
    cols = deleteat!(cols, cols .== "Y")
    cols = deleteat!(cols, cols .== "obj")

    if length(cols) == 0
        println("No observed confounders or covariates found in data")
        X = nothing
    else
        X_ = df[!, cols]
        nX = size(X_)[2]
        X = zeros(n, nX)
        for i in 1:nX
            X[:, i] = Vector(X_[!, i])
        end
    end

    return SigmaU, obj, X, T, Y
end