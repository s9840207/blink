from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_scatter_and_linear_fit(
    x,
    y,
    fit_result: dict,
    save_fig_path: Path = None,
    x_label: str = "",
    y_label: str = "",
    left_bottom_as_origin=False,
    y_top=None,
    x_right=None,
):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    line_x = np.linspace(min(x), max(x), 10)
    line_y = line_x * fit_result["slope"] + fit_result["intercept"]
    ax.plot(line_x, line_y)
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input("Enter x axis label:\n")
    if not y_label:
        y_label = input("Enter y axis label:\n")
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.text(
        0.1,
        0.9,
        r"$y = {:.7f}x + {:.5f}$".format(fit_result["slope"], fit_result["intercept"]),
        transform=ax.transAxes,
        fontsize=12,
    )
    ax.text(
        0.1,
        0.8,
        r"$R^2 = {:.5f}$".format(fit_result["r_squared"]),
        transform=ax.transAxes,
        fontsize=12,
    )
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(save_fig_path, format="svg", dpi=300, bbox_inches="tight")


def plot_error_and_linear_fit(
    x,
    y,
    y_err,
    fit_result: dict,
    save_fig_path: Path = None,
    x_label: str = "",
    y_label: str = "",
    left_bottom_as_origin=False,
    y_top=None,
    x_right=None,
    x_raw=None,
    y_raw=None,
):
    plt.style.use(str(Path(__file__).parent / "fig_style.mplstyle"))
    # sns.set_palette(palette='muted')
    np.random.seed(0)
    fig, ax = plt.subplots()

    sns.despine(fig, ax)
    # ax.set_xticks(sorted(x))
    ax.errorbar(x, y, yerr=y_err, marker="o", ms=2.5, linestyle="")
    line_x = np.array([min(x), max(x)])
    line_y = line_x * fit_result["slope"] + fit_result["intercept"]
    ax.plot(line_x, line_y, zorder=0)
    # color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    x_jitter = (
        0.01 * (x_raw.max() - x_raw.min()) * np.random.standard_normal(x_raw.shape)
    )
    ax.scatter(
        x=x_raw + x_jitter,
        y=y_raw,
        marker="o",
        color="w",
        edgecolors="gray",
        linewidth=0.5,
        s=5,
        zorder=3,
    )
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input("Enter x axis label:\n")
    if not y_label:
        y_label = input("Enter y axis label:\n")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.text(0.05, 0.8, r'$R^2 = {:.5f}$'.format(fit_result['r_squared']),
    #         transform=ax.transAxes, fontsize=8)
    # plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_fig_path, format="svg")


def plot_error(
    x,
    y,
    y_err,
    save_fig_path: Path = None,
    x_label: str = "",
    y_label: str = "",
    left_bottom_as_origin=False,
    y_top=None,
    x_right=None,
    x_raw=None,
    y_raw=None,
):
    plt.style.use(str(Path(__file__).parent / "fig_style.mplstyle"))
    np.random.seed(0)
    fig, ax = plt.subplots()

    sns.despine(fig, ax)
    ax.errorbar(x, y, yerr=y_err, marker="o", ms=2.5, linestyle="")
    x_jitter = (
        0.01 * (x_raw.max() - x_raw.min()) * np.random.standard_normal(x_raw.shape)
    )
    ax.scatter(
        x=x_raw + x_jitter,
        y=y_raw,
        marker="o",
        color="w",
        edgecolors="gray",
        linewidth=0.5,
        s=5,
        zorder=3,
    )
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input("Enter x axis label:\n")
    if not y_label:
        y_label = input("Enter y axis label:\n")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(save_fig_path, format="svg")
