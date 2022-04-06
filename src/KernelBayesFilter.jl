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

function compute_gamma(U, X, KX, ϵ)
    n = size(X)[2]
    GX = kernelmatrix(KX, X)
    m = mean(kernelmatrix(KX, X, U); dims=2)
    γ = n * ((GX + n * ϵ * I(n)) \ m)
    return γ[:, 1]
end

function kernel_bayes_rule(
    U,
    X,
    Y;
    ϵ=0.2,
    KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)),
    KY=with_lengthscale(SqExponentialKernel(), median_lengthscale(Y)),
)::Function
    n = size(X)[2]
    GY = kernelmatrix(KY, Y)
    γ = compute_gamma(U, X, KX, ϵ)
    D = Diagonal(γ)
    A = D * GY
    R = A * ((A * A + ϵ * I(n)) \ D)
    # posterior mean
    f(Ytest) = X * R * kernelmatrix(KY, Y, Ytest)
    return f
end

function iw_kernel_bayes_rule(
    U,
    X,
    Y;
    ϵ=0.2,
    KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)),
    KY=with_lengthscale(SqExponentialKernel(), median_lengthscale(Y)),
)::Function
    n = size(X)[2]
    GY = kernelmatrix(KY, Y)
    γ = compute_gamma(U, X, KX, ϵ)
    D = Diagonal(sqrt.(max.(γ, 0.0))) / n
    R = D * ((D * GY * D + ϵ * I(n)) \ D)
    # posterior mean
    f(Ytest) = X * R * kernelmatrix(KY, Y, Ytest)
    return f
end

export median_lengthscale, kernel_bayes_rule, iw_kernel_bayes_rule

end
