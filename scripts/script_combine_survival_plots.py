from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLORS = ["#a09b00", "#6572fe", "#d30055"]
COLORS = ["#9671c3", "#69a75f", "#cc5366", "#be883d"]
FILL_COLORS = ["#e2e1b2", "#d0d4fe", "#f1b2cc", ""]


def main():
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    datapath = Path(
        "/mnt/data/Research/PriA_project/analysis_result/20200825/20200825imscroll/"
    )
    filestr = ["L1", "L2", "L3", "L4"]
    num = [62.5, 125, 250, 500]
    sorting = np.argsort(num)
    labels = ["62.5 pM", "125 pM", "250 pM", "500 pM"]
    filestr = [filestr[i] for i in sorting]
    labels = [labels[i] for i in sorting]
    fig, ax = plt.subplots()

    for file, label, color, fill_color in zip(
        filestr,
        labels,
        sns.color_palette(palette="muted"),
        sns.color_palette(palette="pastel"),
    ):
        filepath = datapath / "{}_first_on.npz".format(file)
        npz_file = np.load(filepath)

        time = npz_file["survival_curve"]["time"]
        surv = npz_file["survival_curve"]["surv"]
        # upper_ci = npz_file["survival_curve"]["upper_ci"]
        # lower_ci = npz_file["survival_curve"]["lower_ci"]

        def S(t, k1, k2, A):
            return A * np.exp(-k1 * t) + (1 - A) * np.exp(-k2 * t)

        param = npz_file["param"]
        x = np.linspace(0, time[-1], int(round(time[-1] * 10)))
        y = S(x, *param)

        ax.step(time, surv, where="post", color=color, label=label)
        ax.plot(x, y, color=color)
        fill_color = fill_color + (80 / 255,)
        # ax.fill_between(time, lower_ci, upper_ci, step='post', color=fill_color)
    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, time[-1]))
    ax.set_xlim((0, 500))
    ax.set_xlabel(r"$t_{\mathrm{on}}$ (s)")
    ax.set_ylabel("Survival probability")
    ax.legend(frameon=False)
    fig.savefig(datapath / "temp.svg", format="svg")


if __name__ == "__main__":
    main()
