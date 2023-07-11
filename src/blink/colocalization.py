from tqdm import tqdm

import blink.image_processing as imp
import blink.time_series as tseries


def calculate_colocalization_time_traces(
    image_group,
    frame_range,
    target_channel,
    binder_channels_map,
    aois,
    drifter,
    thresholds,
):
    channel_times = {channel: sequence.time for channel, sequence in image_group}
    traces = tseries.TimeTraces(n_traces=len(aois), channels=channel_times)
    for i, image in zip(
        frame_range[target_channel],
        image_group.sequences[target_channel][frame_range[target_channel]],
    ):
        time = image_group.sequences[target_channel].time[i]
        drifted_aois = drifter.shift_aois(aois, i)
        intensity = drifted_aois.get_intensity(image)
        traces.set_value(
            "raw_intensity", channel=target_channel, time=time, array=intensity
        )
        background = drifted_aois.get_background_intensity(image)
        traces.set_value(
            "background", channel=target_channel, time=time, array=background
        )
        traces.set_value(
            "intensity", channel=target_channel, time=time, array=intensity - background
        )

    for channel, mapper in binder_channels_map.items():
        channel_obj = imp.Channel(channel, channel)
        ref_aoi_high = []
        ref_aoi_low = []
        aois_list = []
        mapped_aois = mapper.map(aois, to_channel=channel)
        time_list = []
        for i, image in tqdm(
            zip(
                frame_range[channel_obj],
                image_group.sequences[channel_obj][frame_range[channel_obj]],
            ),
            total=image_group.sequences[channel_obj].length,
        ):
            time = image_group.sequences[channel_obj].time[i]
            time_list.append(time)
            drifted_aois = drifter.shift_aois(mapped_aois, i)
            aois_list.append(drifted_aois)
            intensity = drifted_aois.get_intensity(image)
            traces.set_value(
                "raw_intensity", channel=channel_obj, time=time, array=intensity
            )
            background = drifted_aois.get_background_intensity(image)
            traces.set_value(
                "background", channel=channel_obj, time=time, array=background
            )
            traces.set_value(
                "intensity",
                channel=channel_obj,
                time=time,
                array=intensity - background,
            )
            ref_aoi_high.append(
                imp.pick_spots(image, threshold=thresholds[channel_obj][0])
            )
            ref_aoi_low.append(
                imp.pick_spots(image, threshold=thresholds[channel_obj][1])
            )
        is_colocalized = imp._colocalization_from_high_low_spots(
            aois_list, ref_aoi_high, ref_aoi_low
        )
        traces.set_is_colocalized(channel_obj, is_colocalized, time_array=time_list)
    return traces
