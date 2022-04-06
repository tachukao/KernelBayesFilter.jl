module Run
using Distributions
using Plots
using LinearAlgebra
using KernelBayesFilter

function mse(x, y)
    return mean(sum((x - y) .^ 2; dims=1))
end

function toy()
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
    methods = [kbr, iwkbr]
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
    xlabel!("dimensionality")
    display(fig)

    return nothing
end

end
