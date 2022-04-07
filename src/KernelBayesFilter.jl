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

function propagate(weights, U, X, KX, ϵ)
    n = size(X)[2]
    GX = kernelmatrix(KX, X)
    m = kernelmatrix(KX, X, U) * weights
    new_weights = ((GX + n * ϵ * I(n)) \ m)
    return new_weights
end

function original_update(γ, Y, KY; ϵ, GY=kernelmatrix(KY, Y))
    n = size(Y)[2]
    D = Diagonal(γ[:])
    A = D * GY
    R = A * ((A * A + ϵ * I(n)) \ D)
    # posterior mean
    f(Ytest) = R * kernelmatrix(KY, Y, Ytest)
    return f
end

function iw_update(γ, Y, KY; ϵ, GY=kernelmatrix(KY, Y))
    n = size(Y)[2]
    D = Diagonal(sqrt.(max.(γ[:], 0.0)))
    R = D * ((D * GY * D + ϵ * I(n)) \ D)
    # posterior mean
    f(Ytest) = R * kernelmatrix(KY, Y, Ytest)
    return f
end

function select_update(method)
    if method == :original
        return original_update
    elseif method == :iw
        return iw_update
    else
        throw(error("Method must be either :original or :iw"))
    end
end

"""
    Kernel Bayes' Rule
"""
function kbr(
    wu,
    U,
    X,
    Y;
    ϵ=0.2,
    KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)),
    KY=with_lengthscale(SqExponentialKernel(), median_lengthscale(Y)),
    method=:original,
)
    γ = propagate(wu, U, X, KX, ϵ)
    return select_update(method)(γ, Y, KY; ϵ)
end

function _transition_kernel(
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

"""
    Kernel Bayes' Filter
"""
function kbf(
    X,
    Y,
    Ytest;
    ϵ,
    δ,
    KX=with_lengthscale(SqExponentialKernel(), median_lengthscale(X)),
    KY=with_lengthscale(SqExponentialKernel(), median_lengthscale(Y)),
    method=:original,
)
    T = size(Y)[2]
    Ttest = size(Ytest)[2]
    update = select_update(method)

    P = _transition_kernel(X; ϵ, KX)
    GY = kernelmatrix(KY, Y[:, 1:(T - 1)])
    Xf = zeros(size(X)[1], Ttest)
    kyt = kernelmatrix(KY, Y[:, 1:(T - 1)], Ytest[:, [1]])
    α = (GY + (T - 1) * ϵ * I(T - 1)) \ kyt
    Xf[:, 1] = X[:, 1:(T - 1)] * α

    for t in 2:Ttest
        μ = P * α
        α = update(μ, Y[:, 1:(T - 1)], KY; ϵ=δ, GY)(Ytest[:, [t]])
        Xf[:, t] = X[:, 1:(T - 1)] * α
    end

    return Xf
end

export median_lengthscale, kbr, kbf

end
