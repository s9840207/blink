from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#a09b00", "#6572fe", "#d30055"]
COLORS = ["#9671c3", "#69a75f", "#cc5366", "#be883d"]
FILL_COLORS = ["#e2e1b2", "#d0d4fe", "#f1b2cc", ""]


def main():
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    datapath = Path(
        "/mnt/data/Research/PriA_project/analysis_result/20200922/20200922imscroll/"
    )
    filestr = "L1"
    fig, ax = plt.subplots()

    filepath = datapath / "{}_off.npz".format(filestr)
    npz_file = np.load(filepath)

    time = npz_file["survival_curve"]["time"]
    surv = npz_file["survival_curve"]["surv"]
    upper_ci = npz_file["survival_curve"]["upper_ci"]
    lower_ci = npz_file["survival_curve"]["lower_ci"]
    S = lambda t, k: np.exp(-k * t)
    param = npz_file["param"]
    x = np.linspace(0, time[-1], int(round(time[-1] * 10)))
    y = S(x, *param)

    ax.step(time, surv, where="post")
    ax.plot(x, y, color=plt.rcParams["axes.prop_cycle"].by_key()["color"][2])
    fill_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0].lstrip("#")
    fill_color = tuple(int(fill_color[i : i + 2], 16) / 256 for i in (0, 2, 4)) + (
        80 / 256,
    )
    ax.fill_between(time, lower_ci, upper_ci, step="post", color=fill_color)

    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, time[-1]))
    ax.set_xlabel(r"$t_{\mathrm{off}}$ (s)")
    ax.set_ylabel("Survival probability")
    fig.savefig(datapath / "temp.svg", format="svg")


if __name__ == "__main__":
    main()
