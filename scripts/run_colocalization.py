import blink.drift_correction as dcorr
import blink.image_processing as imp
import gui.image_view as iv
from blink import colocalization, mapping, utils


def main():
    data_dir = iv.select_directory_dialog()
    map_dir = iv.select_directory_dialog()
    image_group_dir = iv.select_directory_dialog()
    image_group = imp.ImageGroup(image_group_dir)
    parameter_file_path = iv.open_file_path_dialog()
    parameters = utils.read_excel(parameter_file_path)
    channels_data = utils.read_excel(parameter_file_path, sheet_name="channels")
    channels_data = channels_data.sort_values(by="order").loc[
        :, ["name", "map file name"]
    ]
    target_channel = imp.Channel(channels_data.name[0], channels_data.name[0])
    binder_channels_map = {
        row["name"]: mapping.Mapper.from_imscroll(
            map_dir / (row["map file name"] + ".dat")
        )
        for i, row in channels_data.iloc[1:, :].iterrows()
    }
    for _, parameter in parameters.iterrows():
        aoiinfo_path = data_dir / (parameter.filename + "_aoi.dat")
        aois = imp.Aois.from_imscroll_aoiinfo2(aoiinfo_path)
        aois.channel = target_channel[1]

        drift_fit_path = data_dir / (parameter.filename + "_driftfit.dat")
        if drift_fit_path.is_file():
            drifter = dcorr.DriftCorrector.from_imscroll_driftfit(drift_fit_path)
        else:
            drifter = dcorr.DriftCorrector(None)

        thresholds = {
            imp.Channel("green", "green"): (parameter.iloc[4], parameter.iloc[5])
        }
        frame_range_value = range(
            int(parameter["framestart"] - 1), int(parameter["frame end"])
        )
        frame_range = {channel: frame_range_value for channel in image_group.channels}
        traces = colocalization.calculate_colocalization_time_traces(
            image_group,
            frame_range,
            target_channel,
            binder_channels_map,
            aois,
            drifter,
            thresholds,
        )
        traces.to_npz(data_dir / (parameter.filename + "_traces.npz"))


if __name__ == "__main__":
    main()
