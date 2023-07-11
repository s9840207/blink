from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns


def main():
    plt.style.use(str(Path("./fig_style.mplstyle").resolve()))
    base_dir = Path("/home/tzu-yu/Downloads/LYT/Cdc13-WT 5 nM")
    data = np.load(base_dir / "msm_result.npz")
    fig, ax = plt.subplots()
    time = np.arange(data["observed_prevalence"].shape[0]) * 0.05
    for state in range(3):
        ax.plot(
            time,
            data["observed_prevalence"][:, state],
            color=sns.color_palette("muted")[state],
            label=f"State {state+1}",
        )
        ax.plot(
            time,
            data["expected_prevalence"][:, state],
            color=sns.color_palette("muted")[state],
            linestyle="--",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Prevalence (%)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(loc="upper right", frameon=False, fontsize=5)
    fig.savefig(base_dir / "prevalence.svg", format="svg")


if __name__ == "__main__":
    main()
