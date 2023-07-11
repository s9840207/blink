from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import blink.time_series as ts
import blink.binding_kinetics as bk
import blink.image_processing as imp


def get_raster_data(category, traces, channel, thresh, time_zero=0):
    y = []
    xmin = []
    xmax = []
    first_event_start = []
    time = traces.get_time(channel)-time_zero
    molecule_idx = []
    for molecule, analyzable in enumerate(category):
        # if not analyzable:
        #     continue

        binary_trace = traces.get_intensity(channel, molecule) > thresh
        state_slices = list(bk.get_interval_slices(binary_trace))
        first_event = True
        for state_slice in state_slices:
            start = state_slice.start
            end = state_slice.stop - 1
            if binary_trace[start] and binary_trace[end]:
                y.append(molecule)
                xmin.append(time[start])
                xmax.append(time[end])
                if first_event:
                    molecule_idx.append(molecule)
                    first_event_start.append(start)
                    first_event = False
    index = np.argsort(first_event_start)
    idx_map = {molecule_idx[j]: i for i, j in enumerate(index)}
    y = [idx_map[i] for i in y]
    
    
    return y, xmin, xmax


def main():
    datadir = Path(r"D:\CWH\2023\20230703\2_g 10min real time_aoi") # aoi folder
    filestr = "g_combined" # name of aoi file
    first_bright_frame = 0
    channel = imp.Channel("red", "red")
    image_group = imp.ImageGroup(Path(r'D:\CWH\2023\20230703\2_r 10min real time'))# image folder
    time_zero = image_group.sequences[channel].time[first_bright_frame]
    traces = ts.TimeTraces.from_npz(datadir / f"{filestr}_traces.npz")
    category_path = (datadir / (filestr + "_category.npy"))
    category = np.load(category_path)
   
    thresh = 6000
    y, xmin, xmax = get_raster_data(category, traces, channel, thresh, time_zero=time_zero)
    plt.style.use(str(Path("./src/blink/fig_style.mplstyle").resolve()))
    fig, ax = plt.subplots()
    ax.hlines(y, xmin, xmax, color=sns.color_palette()[2], linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Molecule index")
    ax.set_yticks([0, traces.n_traces])

    time_offset = traces.get_time(channel)[0] -time_zero
    ax.fill(
        [0, time_offset, time_offset, 0],
        [ax.get_ylim()[0], ax.get_ylim()[0], ax.get_ylim()[1], ax.get_ylim()[1]],
        "#e4e4e4",
    )
    
    fig.savefig(datadir / f"{filestr}_raster.png", format="png", dpi=1200)


def cm2inch(*tupl):
    """
    stackoverflow 14708695
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def raster_BCDX2_dependence_RAD51():
    plt.style.use(str(Path("./src/blink/fig_style.mplstyle").resolve()))
    base_dir = Path("~/analysis_results/").expanduser()
    conds = [
        ("20211020", "L2", 0),
        ("20211020", "L3", 500),
        ("20211029", "L2_01", 62.5),
        ("20211029", "L3_02", 31.25),
        ("20211029", "L4_01", 125),
    ]
    ind = np.argsort([conc for _, _, conc in conds])
    conds = [conds[i] for i in ind]
    channel = imp.Channel("red", "red")
    thresh = 600
    fig, axes = plt.subplots(ncols=len(conds), figsize=cm2inch(13.7 * 1.5, 3 * 1.5))
    linewidth_factor = 0.5 * 150
    for (date, filestr, conc), ax in zip(conds, axes):
        traces = ts.TimeTraces.from_npz(
            base_dir / f"{date}/{date}imscroll/{filestr}_traces.npz"
        )
        y, xmin, xmax = get_raster_data(traces, channel, thresh)
        t0 = traces.get_time(("green", "green"))[0]
        ax.hlines(
            y,
            xmin - t0,
            xmax - t0,
            color=sns.color_palette()[0],
            linewidth=linewidth_factor / traces.n_traces,
        )
        ax.set_xlabel("Time (s)")
        ax.set_yticks([0, traces.n_traces])
        ax.set_xlim(0, 1500)
        ax.set_title(f"{conc} nM")
    axes[0].set_ylabel("Molecule index")
    fig.savefig(Path("~/raster.png").expanduser(), dpi=1200)


def raster_BCDX2_dependence_RAD51_separate():
    plt.style.use(str(Path("./src/blink/fig_style.mplstyle").resolve()))
    base_dir = Path("~/analysis_results/").expanduser()
    conds = [
        ("20211020", "L2", 0),
        ("20211020", "L3", 500),
        ("20211029", "L2_01", 62.5),
        ("20211029", "L3_02", 31.25),
        ("20211029", "L4_01", 125),
    ]
    ind = np.argsort([conc for _, _, conc in conds])
    conds = [conds[i] for i in ind]
    channel = imp.Channel("red", "red")
    thresh = 600
    # fig, axes = plt.subplots(ncols=len(conds), figsize=cm2inch(13.7 * 1.5, 3 * 1.5))
    fig, ax = plt.subplots()
    linewidth_factor = 0.5 * 150
    for (date, filestr, conc) in conds:
        traces = ts.TimeTraces.from_npz(
            base_dir / f"{date}/{date}imscroll/{filestr}_traces.npz"
        )
        y, xmin, xmax = get_raster_data(traces, channel, thresh)
        t0 = traces.get_time(("green", "green"))[0]
        ax.hlines(
            y,
            xmin - t0,
            xmax - t0,
            color=sns.color_palette()[0],
            linewidth=linewidth_factor / traces.n_traces,
        )
        ax.set_xlabel("Time (s)")
        ax.set_yticks([0, traces.n_traces])
        ax.set_xlim(0, 1500)
        ax.set_title(f"{conc} nM")
        ax.set_ylabel("Molecule index")
        fig.savefig(
            Path(
                f"~/git_repos/master_thesis/img/elements/BCDX2_RAD51_{conc}nM_raster.png"
            ).expanduser(),
            dpi=1200,
        )
        ax.cla()


def raster_Dylight_SA():
    plt.style.use(str(Path("./src/blink/fig_style.mplstyle").resolve()))
    base_dir = Path("~/analysis_results/").expanduser()
    conds = [
        ("20211020", "L1", 250),
    ]
    ind = np.argsort([conc for _, _, conc in conds])
    conds = [conds[i] for i in ind]
    channel = imp.Channel("red", "red")
    thresh = 600
    # fig, axes = plt.subplots(ncols=len(conds), figsize=cm2inch(13.7 * 1.5, 3 * 1.5))
    fig, ax = plt.subplots()
    linewidth_factor = 0.5 * 150
    for (date, filestr, conc) in conds:
        traces = ts.TimeTraces.from_npz(
            base_dir / f"{date}/{date}imscroll/{filestr}_traces.npz"
        )
        y, xmin, xmax = get_raster_data(traces, channel, thresh)
        t0 = traces.get_time(("green", "green"))[0]
        ax.hlines(
            y,
            xmin - t0,
            xmax - t0,
            color=sns.color_palette()[0],
            linewidth=linewidth_factor / traces.n_traces,
        )
        ax.set_xlabel("Time (s)")
        ax.set_yticks([0, traces.n_traces])
        ax.set_xlim(0, 1500)
        ax.set_title(f"{conc} nM")
        ax.set_ylabel("Molecule index")
        fig.savefig(
            Path(f"~/Research/DylightSA_RAD51_{conc}nM_raster.png").expanduser(),
            dpi=1200,
        )
        ax.cla()


def raster_250():
    plt.style.use(str(Path("./src/blink/fig_style.mplstyle").resolve()))
    base_dir = Path("~/analysis_results/").expanduser()
    conds = [
        ("20211015", "L2", 250),
    ]
    ind = np.argsort([conc for _, _, conc in conds])
    conds = [conds[i] for i in ind]
    channel = imp.Channel("red", "red")
    thresh = 600
    # fig, axes = plt.subplots(ncols=len(conds), figsize=cm2inch(13.7 * 1.5, 3 * 1.5))
    fig, ax = plt.subplots()
    linewidth_factor = 0.5 * 150
    for (date, filestr, conc) in conds:
        traces = ts.TimeTraces.from_npz(
            base_dir / f"{date}/{date}imscroll/{filestr}_traces.npz"
        )
        y, xmin, xmax = get_raster_data(traces, channel, thresh)
        t0 = traces.get_time(("green", "green"))[0]
        ax.hlines(
            y,
            xmin - t0,
            xmax - t0,
            color=sns.color_palette()[0],
            linewidth=linewidth_factor / traces.n_traces,
        )
        ax.set_xlabel("Time (s)")
        ax.set_yticks([0, traces.n_traces])
        ax.set_xlim(0, 1500)
        ax.set_title(f"{conc} nM")
        ax.set_ylabel("Molecule index")
        fig.savefig(
            Path(f"~/Research/BCDX2_{conc}nM_raster.png").expanduser(),
            dpi=1200,
        )
        ax.cla()


if __name__ == "__main__":
    # raster_BCDX2_dependence_RAD51_separate()
    # raster_Dylight_SA()
    # raster_250()
    main()