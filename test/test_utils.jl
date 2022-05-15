
"""Count how many of the actual mean values are within the expected bounds"""
function countCloseEnough(expected, actual)
    @assert size(expected) == size(actual) "expected and actual aren't same size"
    n = size(actual, 1)
    passing = zeros(n)
    for i = 1:n
        passing[i] = expected[i, "LowerBound"] <= actual[i, "Mean"] &&
                     actual[i, "Mean"] <= expected[i, "UpperBound"]
    end
    return sum(passing) / n
end