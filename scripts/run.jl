module Run
using Distributions
using Plots
using LinearAlgebra
using KernelBayesFilter
using KernelFunctions

function mse(x, y)
    return mean(sum((x - y) .^ 2; dims=1))
end

function toy()
    """
        Reproducing Figure 1 in https://arxiv.org/pdf/2202.02474.pdf
    """

    function problem(d)
        n = 2 * d

        # construct distributions
        A = randn(n, n)
        Σ = ((A' * A) / n) + (2.0 * I(n))
        μ = [zeros(d); ones(d)]
        joint = MvNormal(μ, Σ)
        prior = MvNormal(μ[1:d], 0.5 * Σ[1:d, 1:d])

        # build function to compute posterior mean
        Vxx = Σ[1:d, 1:d]
        Vyy = Σ[(d + 1):end, (d + 1):end]
        Vyx = Σ[(d + 1):end, 1:d]
        Z = Vxx \ Vyx'
        f(y) = Z * ((Vyy - 0.5 * Vyx * Z) \ (y .- 1.0))

        # draw data
        n_prior_samples = 200
        n_train_samples = n_prior_samples
        # prior weights
        wu = ones(n_prior_samples) / n_prior_samples
        # prior samples
        U = rand(prior, n_prior_samples)
        D = rand(joint, n_train_samples)
        X = D[1:d, :]
        Y = D[(d + 1):end, :]
        n_test_samples = 1000
        Dtest = rand(joint, n_test_samples)
        Ytest = Dtest[(d + 1):end, :] .- 1.0
        target = f(Ytest)

        return wu, U, X, Y, Ytest, target
    end

    function experiment(d, method)
        wu, U, X, Y, Ytest, target = problem(d)
        inferred_weights = kbr(wu, U, X, Y; ϵ=0.2, method)(Ytest)
        inferred = X * inferred_weights
        return mse(inferred, target)
    end

    ds = 2 .^ (1:6)
    methods = [:original, :iw]
    n_runs = 50

    function try_method(method)
        wrapper(d) = experiment(d, method)
        return reduce(hcat, map((x) -> map(wrapper, ds), 1:n_runs))
    end

    errs = map(try_method, methods)

    fig = plot(; legend=:topleft, size=(400, 400))
    scatter!(ds, mean(errs[1]; dims=2); ribbon=std(errs[1]; dims=2), label="original")
    scatter!(ds, mean(errs[2]; dims=2); ribbon=std(errs[2]; dims=2), label="IW")
    ylabel!("MSE")
    xlabel!("dimensionality")
    display(fig)

    return nothing
end

struct DynamicParams
    η::Float64
    M::Float64
    β::Float64
end
function DynamicParams(; η, M, β)
    return DynamicParams(η, M, β)
end

function synthetic(method=:original)
    """
        Reproducing results in section 5.3 of Kernel Bayes' Rule
        https://arxiv.org/pdf/1009.5736.pdf
    """
    function f(p::DynamicParams, x)
        u = x[1]
        v = x[2]
        denom = sqrt(u^2 + v^2)
        c = u / denom
        s = v / denom
        nc = cos(p.η) * c - sin(p.η) * s
        ns = sin(p.η) * c + cos(p.η) * s

        rotation = [nc -ns; ns nc]
        coef = p.β * (rotation^p.M)[2, 1]
        nx = (1 + coef) * [nc; ns]
        return nx
    end

    function observe(x, σy)
        return x + σy * randn(size(x))
    end

    function dynamics(p::DynamicParams, x, σx)
        return f(p, x) + σx * randn(size(x))
    end

    function sample(p::DynamicParams, x0, T; σx, σy)
        xs = zeros(length(x0), T)
        ys = zeros(length(x0), T)
        xs[:, 1] = x0
        ys[:, 1] = observe(x0, σy)
        x = x0
        for t in 2:T
            x = dynamics(p, x, σx)
            xs[:, t] = x
            ys[:, t] = observe(x, σy)
        end
        return xs, ys
    end

    params1 = DynamicParams(; η=0.3, M=1.0, β=0.0)
    demo_params1 = DynamicParams(; η=0.05, M=1.0, β=0.0)
    params2 = DynamicParams(; η=0.4, M=8.0, β=0.4)
    demo_params2 = DynamicParams(; η=0.05, M=8.0, β=0.4)

    params = params2
    demo_params = demo_params2

    x0 = [1.0; 0.0]
    σ = 0.2
    ϵ = 1e-5
    δ = ϵ * 2.0
    Ttrain = 500
    Ttest = 200

    Xtrain, Ytrain = sample(params, x0, Ttrain; σx=σ, σy=σ)
    Xtest, Ytest = sample(params, x0, Ttest; σx=σ, σy=σ)
    Xfiltered = kbf(Xtrain, Ytrain, Ytest; ϵ, δ, method)
    nX, _ = sample(demo_params, x0, Ttest; σx=0.0, σy=0.0)

    fig = plot(; legend=:topleft, size=(450, 450))
    plot!(nX[1, :], nX[2, :]; label="trajectory")
    scatter!(Xfiltered[1, 1:end], Xfiltered[2, 1:end]; label="filtered")
    scatter!(Xtest[1, :], Xtest[2, :]; label="true")
    display(fig)
    println("$(String(method)): MSE=$(mse(Xfiltered, Xtest))")

    return nothing
end

end
