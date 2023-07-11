from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import scipy.optimize
import scipy.stats
import statsmodels.api as sm


def main():
    data = pd.read_excel(Path("~/vivien.xlsx").expanduser())
    y, X = patsy.dmatrices("rate ~ conc", data=data, return_type="dataframe")
    model = sm.WLS(y, X, weights=1.0 / (data["err"] ** 2))
    result = model.fit()
    print(result.summary())

    def f(x, a, b):
        return a * x + b

    def f_no_slope(x, b):
        return b

    popt, pcov = scipy.optimize.curve_fit(
        f, data["conc"], data["rate"], sigma=data["err"]
    )
    rss1 = np.sum((f(data["conc"], *popt) - data["rate"]) ** 2 / data["err"] ** 2)
    popt2, pcov2 = scipy.optimize.curve_fit(
        f_no_slope, data["conc"], data["rate"], sigma=data["err"]
    )
    rss2 = np.sum(
        (f_no_slope(data["conc"], *popt2) - data["rate"]) ** 2 / data["err"] ** 2
    )
    F = (rss2 - rss1) * (len(data["conc"]) - 2) / rss1
    print(len(data["conc"]))
    print(popt)
    print(F)
    F_prob = scipy.stats.f.sf(F, 1, len(data["conc"]) - 2)
    print(F_prob)


if __name__ == "__main__":
    main()
