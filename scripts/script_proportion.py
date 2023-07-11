from pathlib import Path

import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats


def main():
    datapath = Path("/home/tzu-yu/Documents/porportion.csv")
    data = np.loadtxt(datapath, delimiter=",")

    rng = np.random.default_rng(1234)
    param = mle_dirichlet(data)
    n_resample = 10000
    n_samples = data.shape[1]
    estimates = []
    for i in range(n_resample):
        samples = rng.dirichlet(param, size=n_samples).T
        est = mle_dirichlet(samples)
        estimates.append(est / est.sum())
    # print(np.mean(estimates, axis=0))
    # print(np.std(estimates, axis=0))
    bootstrap_diff = estimates - param / param.sum()
    alpha = 0.5
    delta = np.quantile(bootstrap_diff, [1 - alpha / 2, alpha / 2], axis=0)
    mean = param / param.sum()
    ci = mean - delta
    print(mean)
    print(ci)
    result = np.concatenate((mean[np.newaxis, :], ci)).T
    with open(datapath.parent / f"{datapath.stem}_dirichlet_bootstrap.csv", "w") as f:
        f.write(",".join(["mean", ".95 CI lower", ".95 CI upper"]))
        f.write("\n")
        np.savetxt(f, result, delimiter=",")


def mle_dirichlet(data):
    rhs = np.log(data).mean(axis=1)

    def f(x):
        return (
            scipy.special.digamma(np.exp(x))
            - scipy.special.digamma(np.exp(x).sum())
            - rhs
        )

    def jac(x):
        jac_alpha = np.diag(
            scipy.special.polygamma(1, np.exp(x))
        ) - scipy.special.polygamma(1, np.exp(x).sum())
        jac = jac_alpha * np.exp(x)[np.newaxis, :]
        return jac

    ini_alpha = rhs
    eps = np.finfo(float).eps
    ln_alpha = scipy.optimize.least_squares(
        f, rhs, method="trf", ftol=eps, xtol=eps, gtol=eps
    ).x
    np.testing.assert_array_less(np.abs(f(ln_alpha)), np.full(ln_alpha.shape, 2e-14))

    alpha = np.exp(ln_alpha)
    return alpha


if __name__ == "__main__":
    main()
