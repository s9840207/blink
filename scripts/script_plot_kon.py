#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import python_for_imscroll.visualization as vis
import scipy.stats
from python_for_imscroll import fitting, utils


def main():
    path = Path(
        "/mnt/data/Research/PriA_project/analysis_result/20210127/20200806-20210127_conc_compile.xlsx"
    ).expanduser()
    df = utils.read_excel(path)

    conc = df["x"]
    k_obs = df["k_on"]
    k_off = df["k_off"]
    print(np.mean(k_off))
    confidence_level = 0.95
    alpha = 1 - (1 - confidence_level) / 2
    ci = scipy.stats.norm.ppf(alpha) * np.std(k_off, ddof=1)
    print(ci)
    x = np.array(list(set(conc)))
    y = np.array([np.mean(k_obs[conc == i]) for i in x])
    y_err = np.array([np.std(k_obs[conc == i], ddof=1) for i in x])
    fit_result = fitting.main(x, y, y_err)
    # fit_result = fitting.main(conc, k_obs)
    vis.plot_error_and_linear_fit(
        x,
        y,
        y_err,
        fit_result,
        Path("/home/tzu-yu/test_obs.svg"),
        x_label="[PriA] (pM)",
        y_label=r"$k_{\mathrm{obs}}$ (s$^{-1}$)",
        left_bottom_as_origin=True,
        x_raw=conc,
        y_raw=k_obs,
    )
    print(fit_result)

    y = np.array([np.mean(k_off[conc == i]) for i in x])
    y_err = np.array([np.std(k_off[conc == i], ddof=1) for i in x])
    vis.plot_error(
        x,
        y,
        y_err,
        Path("/home/tzu-yu/test_off.svg"),
        x_label="[PriA] (pM)",
        y_label=r"$k_{\mathrm{off}}$ (s$^{-1}$)",
        left_bottom_as_origin=True,
        x_raw=conc,
        y_raw=k_off,
        y_top=0.03,
    )
    print(np.mean(k_off) / fit_result["slope"])


if __name__ == "__main__":
    main()
