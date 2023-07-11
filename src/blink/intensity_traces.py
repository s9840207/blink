from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import blink.drift_correction as dcorr
import blink.image_processing as imp
import blink.time_series as tseries
import gui.image_view as iv
from blink import mapping, utils


def calculate_intensity(image_group, frame_range, mapper, aois, drifter):
    channel_times = {channel: sequence.time for channel, sequence in image_group}
    traces = tseries.TimeTraces(
        n_traces=len(aois), channels=channel_times, frame_range=frame_range
    )

    for channel, sequence in image_group:
        aois_list = []
        mapped_aois = mapper.map(aois, to_channel=channel.em)
        time_list = []
        for i, image in tqdm(
            zip(frame_range[channel], sequence[frame_range[channel]]),
            total=len(frame_range[channel]),
        ):
            time = sequence.time[i]
            time_list.append(time)
            drifted_aois = drifter.shift_aois_by_time(mapped_aois, time)
            aois_list.append(drifted_aois)
            intensity = drifted_aois.get_intensity(image)
            traces.set_value(
                "raw_intensity", channel=channel, time=time, array=intensity
            )
            background = drifted_aois.get_background_intensity(image)
            traces.set_value("background", channel=channel, time=time, array=background)
            traces.set_value(
                "intensity", channel=channel, time=time, array=intensity - background
            )
    return traces


def main():
    data_dir = iv.select_directory_dialog()
    map_dir = iv.select_directory_dialog()
    image_master_dir = iv.select_directory_dialog()
    parameter_file_path = iv.open_file_path_dialog()
    parameters = pd.read_excel(parameter_file_path, dtype={"foldername": str})
    channels_data = pd.read_excel(parameter_file_path, sheet_name="channels")
    map_file_paths = [
        map_dir / (name + ".npz") for name in channels_data["map file name"]
    ]
    target_channel = imp.Channel("red", "red")
    mapper = mapping.Mapper.from_npz(Path(r'D:\mapping_for_cosmos_20230425\map.npz'))
    for _, parameter in parameters.iterrows():
        image_group = imp.ImageGroup(image_master_dir / parameter["foldername"])
        aoiinfo_path = data_dir / (parameter.filename + "_aoi.npz")
        aois = imp.Aois.from_npz(aoiinfo_path)
        # print(aois.channel)
        # aois.channel = target_channel.em

        if np.isnan(parameter["framestart"]):
            frame_range = {
                channel: range(image_group.sequences[channel].length)
                for channel in image_group.channels
            }
        else:
            frame_per_cycle = {
                channel: int(
                    round(
                        image_group.sequences[channel].length
                        / image_group.sequences[target_channel].length
                    )
                )
                for channel in image_group.channels
            }
            start_cycle = int(parameter["framestart"])
            frame_range = {
                channel: range(
                    start_cycle * frame_per_cycle[channel],
                    image_group.sequences[channel].length,
                )
                for channel in image_group.channels
            }

        threshold = parameter["drift_thres"]
        driftlist_path = data_dir / (parameter.filename + "_driftlist.npy")
        # if driftlist_path.is_file():
        #     drifter = dcorr.DriftCorrector.from_npy(driftlist_path)
        # else:
        #     breakpoint()
        #     drifter = dcorr.drift_detection(
        #         image_group.sequences[target_channel],
        #         frame_range[target_channel],
        #         threshold,
        #         aois,
        #     )
        drifter = None
        if drifter is None:
            print(f"{parameter.filename} not drift corrected.")
            drifter = dcorr.DriftCorrector(None)
        else:
            drifter.to_npy(data_dir / (parameter.filename + "_driftlist.npy"))
        traces = calculate_intensity(image_group, frame_range, mapper, aois, drifter)
        traces.to_npz(data_dir / (parameter.filename + "_traces.npz"))


def main_fret():
    data_dir = iv.select_directory_dialog()
    map_dir = Path("/home/tzu-yu/git_repos/Imscroll-and-Utilities/data/mapping/")
    image_master_dir = iv.select_directory_dialog()
    parameter_file_path = iv.open_file_path_dialog()
    parameters = utils.read_excel(parameter_file_path)
    channels_data = utils.read_excel(parameter_file_path, sheet_name="channels")
    map_file_paths = [
        map_dir / (name + ".dat") for name in channels_data["map file name"]
    ]
    mapper = mapping.Mapper.from_imscroll(map_file_paths)
    for _, parameter in parameters.iterrows():
        image_group = imp.ImageGroup(image_master_dir / parameter["foldername"])
        aoiinfo_path = data_dir / (parameter.filename + "_aoi.npz")
        aois = imp.Aois.from_npz(aoiinfo_path)
        aois = combine_spots(mapper, aois)

        n = int(parameter["n"])
        frame_range = {
            imp.Channel("green", "green"): range(n * 200, (n + 1) * 200),
            imp.Channel("green", "red"): range(n * 200, (n + 1) * 200),
            imp.Channel("red", "red"): range(n * 10, (n + 1) * 10),
        }
        drifter = dcorr.DriftCorrector(None)
        traces = calculate_intensity(image_group, frame_range, mapper, aois, drifter)
        traces.to_npz(data_dir / (parameter.filename + "_traces.npz"))


def combine_spots(mapper, aois):
    channel_a = imp.Aois(
        coords=aois.coords[aois.coords[:, 0] >= 512, :],
        frame=aois.frame,
        frame_avg=aois.frame_avg,
        width=aois.width,
        channel="red",
    )
    channel_b = imp.Aois(
        coords=aois.coords[aois.coords[:, 0] < 512, :],
        frame=aois.frame,
        frame_avg=aois.frame_avg,
        width=aois.width,
        channel=aois.channel,
    )
    mapped_a = mapper.map(channel_a, to_channel="green")
    is_in_range = mapped_a.is_in_range_of(channel_b, 3)
    mapped_a.coords = np.concatenate(
        (mapped_a.coords[~is_in_range, :], channel_b.coords), axis=0
    )
    return mapped_a


def main_cy7():
    data_dir = iv.select_directory_dialog()
    image_master_dir = iv.select_directory_dialog()
    parameter_file_path = iv.open_file_path_dialog()
    parameters = pd.read_excel(parameter_file_path, dtype={"foldername": str})
    target_channel = imp.Channel("red", "red")
    mapper = mapping.Mapper()
    mapper.map_matrix = {
        mapping.MapDirection("red", "ir"): np.array([[1, 0, 0], [0, 1, 0]])
    }
    for _, parameter in parameters.iterrows():
        image_group = imp.ImageGroup(image_master_dir / parameter["foldername"])
        cy7_aois_path = data_dir / (parameter.filename + "_ir_aoi.npz")
        cy5_aois_path = data_dir / (parameter.filename + "_r_aoi.npz")
        cy5_aois = imp.Aois.from_npz(cy5_aois_path)
        cy7_aois = imp.Aois.from_npz(cy7_aois_path)

        is_in_range = cy5_aois.is_in_range_of(cy7_aois, 3)
        cy5_aois.coords = np.concatenate(
            (cy5_aois.coords[~is_in_range, :], cy7_aois.coords), axis=0
        )
        aois = cy5_aois
        aois.channel = target_channel.em

        if np.isnan(parameter["framestart"]):
            frame_range = {
                channel: range(image_group.sequences[channel].length)
                for channel in image_group.channels
            }
        else:
            frame_per_cycle = {
                channel: int(
                    round(
                        image_group.sequences[channel].length
                        / image_group.sequences[target_channel].length
                    )
                )
                for channel in image_group.channels
            }
            start_cycle = int(parameter["framestart"])
            frame_range = {
                channel: range(
                    start_cycle * frame_per_cycle[channel],
                    image_group.sequences[channel].length,
                )
                for channel in image_group.channels
            }

        threshold = parameter["drift_thres"]
        driftlist_path = data_dir / (parameter.filename + "_driftlist.npy")
        if driftlist_path.is_file():
            drifter = dcorr.DriftCorrector.from_npy(driftlist_path)
        else:
            drifter = dcorr.drift_detection(
                image_group.sequences[target_channel],
                frame_range[target_channel],
                threshold,
                aois,
            )
            if drifter is None:
                print(f"{parameter.filename} not drift corrected.")
                drifter = dcorr.DriftCorrector(None)
            else:
                drifter.to_npy(data_dir / (parameter.filename + "_driftlist.npy"))
        traces = calculate_intensity(image_group, frame_range, mapper, aois, drifter)
        traces.to_npz(data_dir / (parameter.filename + "_traces.npz"))


if __name__ == "__main__":
    main()
    #main_fret()
