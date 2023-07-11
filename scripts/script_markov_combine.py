from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats


def main():
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    base_dir = Path("/home/tzu-yu/play_dead")
    base_dir = Path("/home/tzu-yu/Downloads/LYT/Cdc13-WT")
    exclude_dir_list = [
        "Cdc13-WT 10 nM_new slide",
        "Cdc13 heterodimer 10 nM_new slide",
        "results",
    ]
    subdir_names = [
        path.stem
        for path in base_dir.iterdir()
        if path.is_dir() and path.stem not in exclude_dir_list
    ]
    concs = [int(name.split()[1]) for name in subdir_names]
    sort_index = np.argsort(concs)
    concs = [concs[i] for i in sort_index]
    subdir_names = [subdir_names[i] for i in sort_index]
    data = dict()
    for conc, subdir_name in zip(concs, subdir_names):
        data[conc] = np.load(base_dir / subdir_name / "msm_result.npz")

    result_dir = base_dir / "results"
    if not result_dir.exists():
        result_dir.mkdir()

    fig, axs = plt.subplots(3, 3, figsize=(6, 4.5))
    for ini_state in range(3):
        for fin_state in range(3):
            if ini_state == fin_state:
                continue
            k = [data[conc]["estimate"][ini_state, fin_state] for conc in concs]
            k_upper = [
                data[conc]["upper_bound"][ini_state, fin_state] for conc in concs
            ]
            err_upper = [k_upper_i - k_i for k_upper_i, k_i in zip(k_upper, k)]
            k_lower = [
                data[conc]["lower_bound"][ini_state, fin_state] for conc in concs
            ]
            err_lower = [k_i - k_lower_i for k_lower_i, k_i in zip(k_lower, k)]

            ln_k = np.log(k)
            ln_k_upper = np.log(k_upper)
            ln_error = ln_k_upper - ln_k

            df = pd.DataFrame(
                {
                    "conc": concs,
                    "k": k,
                    "ln_error": ln_error,
                    ".95 CI upper": k_upper,
                    ".95 CI lower": k_lower,
                }
            )

            def f_association(x, a):
                return np.log(a * x)

            def f_dissociation(x, a, b):
                return np.log(a * x + b)

            def f_no_slope(x, b):
                return np.log(b)

            ax = axs[ini_state][fin_state]
            ax.errorbar(
                concs, k, yerr=[err_lower, err_upper], marker="o", ms=2.5, linestyle=""
            )
            ax.set_xlabel("wtCdc13 concentration (nM)")

            if fin_state > ini_state:
                ini_a = k[-1] / concs[-1]
                popt, pcov = scipy.optimize.curve_fit(
                    f_association, concs, ln_k, p0=ini_a, sigma=ln_error
                )
                x = np.linspace(concs[0], concs[-1])
                ax.plot(x, np.exp(f_association(x, *popt)))
            else:
                ini_a = (k[-1] - k[0]) / (concs[-1] - concs[0])
                ini_b = k[0] - concs[0] * ini_a
                x = np.array(concs)[~np.isinf(ln_k)]
                y = ln_k[~np.isinf(ln_k)]
                err = ln_error[~np.isinf(ln_k)]
                popt, pcov = scipy.optimize.curve_fit(
                    f_dissociation,
                    np.array(concs)[~np.isinf(ln_k)],
                    ln_k[~np.isinf(ln_k)],
                    p0=[ini_a, ini_b],
                    sigma=ln_error[~np.isinf(ln_k)],
                )
                rss1 = np.sum(((f_dissociation(x, *popt) - y) / err) ** 2)
                ini_b = np.mean(k)
                popt2, pcov2 = scipy.optimize.curve_fit(
                    f_no_slope,
                    np.array(concs)[~np.isinf(ln_k)],
                    ln_k[~np.isinf(ln_k)],
                    p0=ini_b,
                    sigma=ln_error[~np.isinf(ln_k)],
                )
                rss2 = np.sum(((f_no_slope(x, *popt2) - y) / err) ** 2)
                F = (rss2 - rss1) * (len(x) - 2) / rss1
                F_prob = scipy.stats.f.sf(F, 1, len(x) - 2)
                x = np.linspace(concs[0], concs[-1])
                ax.plot(x, np.exp(f_dissociation(x, *popt)))
            ax.set_ylabel(f"k ({ini_state+1} -> {fin_state+1})")
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            with open(
                result_dir / f"transition_{ini_state+1}_to_{fin_state+1}.csv", "w"
            ) as f:
                df.to_csv(f, index=False)
                f.write("\n\n")
                f.write("Linear regression result\n")
                if fin_state > ini_state:
                    f.write("slope,{}\n".format(*popt))
                else:
                    f.write("slope,{}\nintercept,{}\n".format(*popt))
                    f.write(f"F statistics (compared to no slope),{F}\n")
                    f.write(f"F prob (compared to no slope),{F_prob}\n")
                    f.write("intercept only model,{}\n".format(*popt2))

    save_fig_path = base_dir / "plot.svg"
    fig.savefig(save_fig_path, format="svg")


def script_10nM():
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    base_dir = Path("/home/tzu-yu/play_dead")
    base_dir = Path("/home/tzu-yu/Downloads/LYT/")
    # subdir_names = [path.stem for path in base_dir.iterdir() if path.is_dir()]
    # concs = [int(name.split()[1]) for name in subdir_names]
    # sort_index = np.argsort(concs)
    # concs = [concs[i] for i in sort_index]
    # subdir_names = [subdir_names[i] for i in sort_index]
    dir_list = [
        "Cdc13-WT 10 nM",
        "Cdc13-WT 10 nM_new slide",
        "Cdc13 heterodimer 10 nM_new slide",
    ]
    concs = ["WT", "WT new slide", "heterodimer"]
    data = dict()
    for conc, subdir_name in zip(concs, dir_list):
        data[conc] = np.load(base_dir / subdir_name / "msm_result.npz")

    fig, axs = plt.subplots(3, 3, figsize=(6, 4.5))
    for ini_state in range(3):
        for fin_state in range(3):
            if ini_state == fin_state:
                continue
            k = [data[conc]["estimate"][ini_state, fin_state] for conc in concs]
            k_upper = [
                data[conc]["upper_bound"][ini_state, fin_state] for conc in concs
            ]
            err_upper = [k_upper_i - k_i for k_upper_i, k_i in zip(k_upper, k)]
            k_lower = [
                data[conc]["lower_bound"][ini_state, fin_state] for conc in concs
            ]
            err_lower = [k_i - k_lower_i for k_lower_i, k_i in zip(k_lower, k)]

            ax = axs[ini_state][fin_state]
            ax.errorbar(
                concs, k, yerr=[err_lower, err_upper], marker="o", ms=2.5, linestyle=""
            )
            ax.set_xlabel("wtCdc13 concentration (nM)")
            ax.set_ylabel(f"k ({ini_state+1} -> {fin_state+1})")
            ax.set_ylim(bottom=0)
            # ax.set_xlim(left=0)
    save_fig_path = base_dir / "10nM.svg"
    fig.savefig(save_fig_path, format="svg")


if __name__ == "__main__":
    main()
    # script_10nM()
