module KernelBayesFilter

using Distributions
using Plots
using LinearAlgebra
using KernelFunctions
using Distances

function median_lengthscale(X, Y)
    return median(pairwise(Euclidean(), X, Y))
end

function median_lengthscale(X)
    return median_lengthscale(X, X)
end

function kbr_common(U, X, Y; ϵ=0.2)
    n = size(X)[2]
    KX = with_lengthscale(SqExponentialKernel(), median_lengthscale(X))
    KY = with_lengthscale(SqExponentialKernel(), median_lengthscale(Y))
    GX = kernelmatrix(KX, X)
    GY = kernelmatrix(KY, Y)
    m = mean(kernelmatrix(KX, X, U); dims=2)
    μ = n * ((GX + n * ϵ * I(n)) \ m)
    return n, KY, GY, μ
end

function kbr(U, X, Y; ϵ=0.2)
    n, KY, GY, μ = kbr_common(U, X, Y; ϵ)
    Λ = Diagonal(μ[:, 1])
    A = Λ * GY
    R = A * ((A * A + ϵ * I(n)) \ Λ)
    # posterior mean
    f(Y2) = X * R * kernelmatrix(KY, Y, Y2)
    return f
end

function iwkbr(U, X, Y; ϵ=0.2)
    n, KY, GY, μ = kbr_common(U, X, Y; ϵ)
    D = Diagonal(sqrt.(max.(μ[:, 1], 0.0))) / n
    R = D * ((D * GY * D + ϵ * I(n)) \ D)
    # posterior mean
    f(Y2) = X * R * kernelmatrix(KY, Y, Y2)
    return f
end

export kbr, iwkbr, median_lengthscale

end
