module Run
using Distributions
using Plots
using LinearAlgebra
using KernelBayesFilter

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
        U = rand(prior, n_prior_samples)
        D = rand(joint, n_train_samples)
        X = D[1:d, :]
        Y = D[(d + 1):end, :]
        n_test_samples = 1000
        Dtest = rand(joint, n_test_samples)
        Ytest = Dtest[(d + 1):end, :] .- 1.0
        target = f(Ytest)

        return U, X, Y, Ytest, target
    end

    function experiment(d, method)
        U, X, Y, Ytest, target = problem(d)
        inferred = method(U, X, Y; ϵ=0.2)(Ytest)
        return mse(inferred, target)
    end

    dimensionalities = 2 .^ (1:6)
    methods = [kernel_bayes_rule, iw_kernel_bayes_rule]
    n_runs = 50

    function try_method(method)
        wrapper(d) = experiment(d, method)
        return reduce(hcat, map((x) -> map(wrapper, dimensionalities), 1:n_runs))
    end

    errs = map(try_method, methods)

    fig = plot(; legend=:topleft)
    scatter!(ds, mean(errs[1]; dims=2); ribbon=std(errs[1]; dims=2), label="original")
    scatter!(ds, mean(errs[2]; dims=2); ribbon=std(errs[2]; dims=2), label="IW")
    ylabel!("MSE")
    xlabel!("dimensionality")
    display(fig)

    return nothing
end

end
