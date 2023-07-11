#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (visualization.py) is part of blink.
#
#  blink is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  blink is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with blink.  If not, see <https://www.gnu.org/licenses/>.

"""Visualization module handles single-molecule fluorescence data visualization"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from blink import time_series


def plot_one_trace(
    traces: "time_series.TimeTraces", molecule: int, moving_average: int = 1
) -> plt.Figure:
    """Take data of one molecule and plot time trajectory.

    Each subplot corresponds to one channel (color).
    Args:
        traces: The time serie data of a set of molecules.
        molecule: The molecule index to plot.

    Returns:
        A matplotlib Figure object containing the plot.
    """
    plt.style.use(str(Path(__file__).parent / "trace_style.mplstyle"))
    channel_list = sorted(traces.get_channels())
    fig, ax_list = plt.subplots(nrows=len(channel_list), sharex=True)
    if isinstance(ax_list, plt.Axes):
        ax_list = [ax_list]
    fig.suptitle("molecule {}".format(molecule), fontsize=7)
    max_time = max([max(traces.get_time(channel)) for channel in channel_list])
    for channel, ax in zip(channel_list, ax_list):
        time = traces.get_time(channel)
        if moving_average > 1:
            kernel = np.ones(moving_average) / moving_average
            time = np.convolve(time, kernel, "valid")
            if traces.has_variable(channel, "intensity"):
                intensity = np.convolve(
                    traces.get_intensity(channel, molecule), kernel, "valid"
                )
                ax.plot(time, intensity, color=channel.em)

            if traces.has_variable(channel, "viterbi_path"):
                viterbi_path = np.convolve(
                    traces.get_viterbi_path(channel, molecule), kernel, "valid"
                )
                ax.plot(time, viterbi_path, color="black")
        else:
            if traces.has_variable(channel, "intensity"):
                intensity = traces.get_intensity(channel, molecule)
                ax.plot(time, intensity, color=channel.em)

            if traces.has_variable(channel, "viterbi_path"):
                viterbi_path = traces.get_viterbi_path(channel, molecule)
                ax.plot(time, viterbi_path, color="black")

        if ax.get_ylim()[0] > 0:
            ax.set_ylim(bottom=0)

    ax.set_xlim((0, max_time))
    ax.set_xlabel("Time (s)")
    fig.text(0.04, 0.4, "Intensity", ha="center", rotation="vertical")
    return fig
