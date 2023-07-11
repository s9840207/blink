from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import blink.drift_correction
import blink.image_processing as imp
from blink import mapping


def quickMinMax(data):
    """
    Estimate the min/max values of *data* by subsampling.
    Returns [(min, max), ...] with one item per channel

    Copied from the pyqtgraph ImageView class
    """
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]

    if data.size == 0:
        return [(0, 0)]
    return (float(np.nanmin(data)), float(np.nanmax(data)))


def main():
    frame = 1
    datapath = Path(r"D:\CWH\20230204\200nM_aoi").expanduser()
    image_path_green = Path(
        r"D:\CWH\20230204\200nM\hwligroup11118"
    )
    image_path = Path(
        r"D:\CWH\20230204\200nM\hwligroup11120"
    )
    filestr = "g"
    # drifter = blink.drift_correction.DriftCorrector.from_npy(
    #     datapath / (filestr + "_driftlist.npy")
    # )

    edgecolor = sns.color_palette("muted")[0]
    image_sequence = imp.ImageSequence(image_path_green)
    aois = imp.Aois.from_npz(datapath / (filestr + "_aoi.npz"))
    aois.channel = "blue"
    print(image_sequence.time[frame])
    fig, ax = plt.subplots(figsize=(2, 2))
    image = image_sequence.get_averaged_image(frame, 10)
    scale = quickMinMax(image)
    ax.imshow(
        image,
        cmap="gray_r",
        vmin=scale[0],
        vmax=scale[1] - 200,
        interpolation="nearest",
        origin="upper",
    )
    origin = np.array([100, 300]) - 0.5  # Offset by 0.5 to the edge of pixel
    size = 100
    ax.set_axis_off()
    coords = aois.coords
    in_range = np.logical_and.reduce(
        (
            coords[:, 0] > origin[0],
            coords[:, 0] < origin[0] + size,
            coords[:, 1] > origin[1],
            coords[:, 1] < origin[1] + size,
        )
    )
    ax.scatter(
        aois.get_all_x()[in_range],
        aois.get_all_y()[in_range],
        marker="s",
        color="none",
        edgecolors=edgecolor,
        linewidth=1,
        s=70,
    )
    ax.text(
        0.05,
        0.95,
        "RAD51",
        color=sns.color_palette("dark")[2],
        fontfamily="arial",
        fontsize=12,
        fontweight="medium",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ax.set_xlim((origin[0], origin[0] + size))
    ax.set_ylim((origin[1], origin[1] + size))
    fig.savefig(
        Path(datapath / "image_temp_green.svg").expanduser(),
        dpi=1200,
        format="svg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    mapping_file_path = Path(
        r"D:\mapping\map.npz")
    # mapping_file_path_2 = Path(
    #     "~/git_repos/Imscroll-and-Utilities/data/mapping/20210208_rb_5.dat"
    # ).expanduser()
    image_sequence = imp.ImageSequence(image_path) 
    # aois = drifter.shift_aois_by_time(aois, image_sequence.time[frame * 10])
    print(image_sequence.time[frame * 10])
    mapper = mapping.Mapper.from_npz(mapping_file_path)
    # mapper2 = mapping.Mapper.from_imscroll(mapping_file_path_2)
    mapped_aois = mapper.map(aois, to_channel="green")
    mapped_aois = mapper.map(mapped_aois, to_channel="blue")

    aois = mapped_aois
    fig, ax = plt.subplots(figsize=(2, 2))
    image = image_sequence.get_averaged_image(frame * 10, 10)
    print(frame * 10)
    scale = quickMinMax(image)
    ax.imshow(
        image,
        cmap="gray_r",
        vmin=scale[0],
        vmax=scale[1] - 500,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_axis_off()
    ax.scatter(
        aois.get_all_x()[in_range],
        aois.get_all_y()[in_range],
        marker="s",
        color="none",
        edgecolors=edgecolor,
        linewidth=0.8,
        s=70,
        linestyle=":",
    )
    ax.text(
        0.05,
        0.95,
        "RAD51",
        color='#2894FF',
        fontfamily="arial",
        fontsize=12,
        fontweight="medium",
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    ref_aoi = imp.pick_spots(image_sequence.get_averaged_image(frame * 10, 10), 20)
    colocalized_aois = aois.remove_aois_far_from_ref(ref_aoi, 1.5)
    origin -= mapper.map_matrix[("blue", "green")][:, 2]
    # origin -= mapper.map_matrix[("red", "blue")][:, 2]
    coords = colocalized_aois.coords
    in_range = np.logical_and.reduce(
        (
            coords[:, 0] > origin[0],
            coords[:, 0] < origin[0] + size,
            coords[:, 1] > origin[1],
            coords[:, 1] < origin[1] + size,
        )
    )
    ax.scatter(
        colocalized_aois.get_all_x()[in_range],
        colocalized_aois.get_all_y()[in_range],
        marker="s",
        color="none",
        edgecolors=edgecolor,
        linewidth=1,
        s=70,
    )
    ax.set_xlim((origin[0], origin[0] + size))
    ax.set_ylim((origin[1], origin[1] + size))
    fig.savefig(
        datapath / "image_temp_red.svg",
        dpi=1200,
        format="svg",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)



if __name__ == "__main__":
    main()


