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
    γ = ((GX + n * ϵ * I(n)) \ m)
    return γ[:, 1]
end

function kernel_bayes(
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

function iw_kernel_bayes(
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

function _prediction_matrix(
    X; KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)), ϵ
)
    T = size(X)[2]
    X1 = X[:, 1:(T - 1)]
    X2 = X[:, 2:T]
    GX = kernelmatrix(KX, X1)
    GX_ = kernelmatrix(KX, X1, X2)
    L = inv(GX + (T - 1) * ϵ * I(T - 1))
    return L * GX_ * L * GX
end

function _posterior(μ, KY, GY, Y, y; δ)
    T = size(Y)[2]
    D = Diagonal(μ)
    A = D * GY
    ky = kernelmatrix(KY, Y, reshape(y, (length(y), 1)))
    R = (A * A + δ * I(T))
    α = A * (R \ (D * ky))
    return α[:, 1]
end

function kernel_bayes_filter(
    X,
    Y,
    Ytest;
    ϵ,
    δ,
    KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)),
    KY=with_lengthscale(SqExponentialKernel(), median_lengthscale(Y)),
)
    T = size(Y)[2]
    Ttest = size(Ytest)[2]
    P = _prediction_matrix(X; ϵ, KX)
    GY = kernelmatrix(KY, Y[:, 1:(T - 1)])
    Xf = zeros(size(X)[1], Ttest)
    ytest1 = reshape(Ytest[:, 1], (size(Ytest)[1], 1))
    kyt = kernelmatrix(KY, Y[:, 1:(T - 1)], ytest1)[:]
    α = (GY + T * ϵ * I(T - 1)) \ kyt
    Xf[:, 1] = X[:, 1:(T - 1)] * α

    for t in 2:Ttest
        μ = P * α
        α = _posterior(μ, KY, GY, Y[:, 1:(T - 1)], Ytest[:, t]; δ)
        Xf[:, t] = X[:, 1:(T - 1)] * α
    end

    return Xf
end

export median_lengthscale, kernel_bayes, iw_kernel_bayes
export kernel_bayes_filter

end
