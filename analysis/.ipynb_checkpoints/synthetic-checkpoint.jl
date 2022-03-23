# n Should be even.
n = 5

function simData(n)
    U = zeros(n)
    X = zeros(n)
    Y = zeros(n)
    IntY = zeros(n)
    
    for i in 1:n
        U[i] = uniform(-3, 3)
        X[i] = normal(U[i], 1)
        Ymean = (0.3 - 0.1 * X[i] + 0.5 * X[i]^2 - 1 * U[i])/5
        Y[i] = normal(Ymean, 0.1)
    end
    return U, X, Y
end

U, X, Y = simData(n)
scatter(X, Y, c=U)
colorbar()
title("Color = U")
xlabel("Treatment")
ylabel("Outcome")
