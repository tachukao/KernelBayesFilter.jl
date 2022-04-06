module Run
using Distributions
using Plots
using LinearAlgebra
using KernelFunctions
using Statistics

function mse(x, y)
    return mean(sum((x - y) .^ 2; dims=1))
end

function euclidean_distance(x, y; ℓ=1.0)
    x = x / ℓ
    y = y / ℓ
    x_norm = sum(x .^ 2; dims=1)
    x_pad = ones(size(x_norm))
    y_norm = sum(y .^ 2; dims=1)
    y_pad = ones(size(y_norm))
    x_ = [-2.0 * x; x_norm; x_pad]
    y_ = [y; y_pad; y_norm]
    d = x_' * y_
    d[d .< 0.0] .= 0.0
    return sqrt.(d)
end

function gaussian_kernel(x, y; ℓ=1.0)
    return exp.(-euclidean_distance(x, y; ℓ) .^ 2)
end

function gaussian_kernel(x; ℓ=1.0)
    K = gaussian_kernel(x, x; ℓ)
    K[diagind(K)] .= 1.0
    return K
end

function median_lengthscale(X, Y)
    return median(euclidean_distance(X, Y))
end

function median_lengthscale(X)
    return median_lengthscale(X, X)
end

function run()
    function problem(d)
        n = 2 * d
        A = randn(n, n)
        Σ = ((A' * A) / n) + (2.0 * I(n))
        μ = [zeros(d); ones(d)]
        joint = MvNormal(μ, Σ)
        prior = MvNormal(μ[1:d], 0.5 * Σ[1:d, 1:d])
        Vxx = Σ[1:d, 1:d]
        Vyy = Σ[(d + 1):end, (d + 1):end]
        Vyx = Σ[(d + 1):end, 1:d]
        Z = Vxx \ Vyx'
        H = Z * inv(Vyy - 0.5 * Vyx * Z)

        # prec = inv(Σ)
        # SXY = prec[1:d, (d + 1):end]
        # SYY = prec[(d + 1):end, (d + 1):end]
        # B = inv(prior.Σ) + (SXY * (SYY \ SXY'))
        # H = -B \ SXY

        f(y) = H * (y .- 1)

        return joint, prior, f
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

    function experiment(d, method)
        k = 200
        n_samples = k
        joint, prior, target = problem(d)
        ϵ = 0.2

        U = rand(prior, k)
        D = rand(joint, n_samples)
        X, Y = D[1:d, :], D[(d + 1):end, :]

        n_test_samples = 1000
        D2 = rand(joint, n_test_samples)
        Y2 = D2[(d + 1):end, :] .- 1.0

        inferred = method(U, X, Y; ϵ)(Y2)
        analytic = target(Y2)
        return mse(inferred, analytic)
    end

    ds = 2 .^ (1:6)
    methods = [kbr, iwkbr, bootstrap]
    n_runs = 50
    errs = map(
        (method) ->
            reduce(hcat, map((x) -> map((d) -> experiment(d, method), ds), 1:n_runs)),
        methods,
    )

    fig = plot(; legend=:topleft)
    scatter!(ds, mean(errs[1]; dims=2); ribbon=std(errs[1]; dims=2), label="original")
    scatter!(ds, mean(errs[2]; dims=2); ribbon=std(errs[2]; dims=2), label="IW")
    ylabel!("MSE")
    xlabel!("d")
    display(fig)

    return nothing
end

end
