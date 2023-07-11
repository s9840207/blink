from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

from blink import image_processing as imp


class TimeTraces:
    def __init__(self, n_traces, channels=None, frame_range=None):
        if channels is None:
            self._data = None
        else:
            self._data = dict.fromkeys(channels.keys())
        self.n_traces = n_traces
        if frame_range is None:
            self._time = channels
        else:
            self._time = {
                channel: [time[i] for i in frame_range[channel]]
                for channel, time in channels.items()
            }
        self.aoi_categories = None

    def get_n_traces(self):
        return len(self._data.AOI)

    def get_channels(self):
        return list(self._data.keys())

    def has_variable(self, channel, variable_name):
        return variable_name in self._data[channel]

    def get_time(self, channel):
        return self._time[channel]

    def get_intensity(self, channel, molecule):
        data = self._data[channel].sel(molecule=molecule)
        return data.intensity.values

    def get_is_colocalized(self, channel, molecule):
        data = self._data[channel].sel(molecule=molecule)
        return data.is_colocalized.values

    def set_value(self, variable_name, channel, time, array):
        if len(array) != self.n_traces:
            raise ValueError(
                f"Input array length ({len(array)}) does "
                f"not match n_traces ({self.n_traces})."
            )
        time_arr = self._time[channel]
        if self._data[channel] is None:
            data = {
                variable_name: (
                    ["molecule", "time"],
                    np.zeros((self.n_traces, len(time_arr))),
                )
            }
            self._data[channel] = xr.Dataset(
                data,
                coords={
                    "molecule": (["molecule"], np.arange(self.n_traces)),
                    "time": (["time"], time_arr),
                },
            )
        elif variable_name not in self._data[channel].keys():
            self._data[channel] = self._data[channel].assign(
                {
                    variable_name: (
                        ["molecule", "time"],
                        np.zeros((self.n_traces, len(time_arr))),
                    )
                }
            )
        self._data[channel][variable_name].loc[dict(time=time)] = array

    def set_is_colocalized(self, channel, array, time_array=None):
        if array.shape[1] == self.n_traces:
            array = array.T
        elif array.shape[0] != self.n_traces:
            raise ValueError(
                f"No dimension of input array (shape: {array.shape})"
                f"matches n_traces ({self.n_traces})."
            )
        if time_array is None:
            time_array = self._time[channel]
        if self._data[channel] is None:
            data = {"is_colocalized": (["molecule", "time"], array)}
            self._data[channel] = xr.Dataset(
                data,
                coords={
                    "molecule": (["molecule"], np.arange(self.n_traces)),
                    "time": (["time"], time_array),
                },
            )
        elif "is_colocalized" not in self._data[channel].keys():
            array = xr.DataArray(
                array,
                dims=["molecule", "time"],
                coords={
                    "molecule": (["molecule"], np.arange(self.n_traces)),
                    "time": (["time"], time_array),
                },
            )
            self._data[channel] = self._data[channel].assign({"is_colocalized": array})

    def get_viterbi_path(self, channel, molecule):
        data = self._data[channel].sel(molecule=molecule)
        return data.viterbi_path.values

    def to_npz(self, path):
        items_to_save = dict()
        for channel in self.get_channels():
            if self._data[channel] is not None:
                items_to_save["{}-{}".format(*channel)] = (
                    self._data[channel].to_dataframe().to_records()
                )
        np.savez_compressed(path, **items_to_save)

    @classmethod
    def from_npz(cls, path):
        data = dict()
        time = dict()
        with np.load(path) as npz_file:
            for file_name in npz_file.files:
                channel = imp.Channel(*file_name.split("-"))
                data_frame = pd.DataFrame(npz_file[file_name]).set_index(
                    ["molecule", "time"]
                )
                data[channel] = xr.Dataset.from_dataframe(data_frame)
                time[channel] = data[channel].time.values
        n_traces = len(data[channel].molecule)
        traces = cls(n_traces)
        traces._data = data
        traces._time = time
        return traces

    def get_index_from_time(self, time: float):
            indices = dict()
            for channel in self.get_channels():
                time_arr = self.get_time(channel)

                if time_arr[0] <= time <= time_arr[-1]:
                    indices[channel] = np.argmin(np.abs(time_arr-time))
            return indices

    # def get_index_from_time(self, time: float):
    #     indices = dict()
    #     for channel in self.get_channels():
    #         time_arr = self.get_time(channel)
    #         if time_arr[0] <= time <= time_arr[-1]:
    #             nearest_time = time_arr.sel(time=time, method="nearest").item()
    #             indices[channel] = np.nonzero(time_arr.values == nearest_time)[0].item()
    #     return indices

    def get_intensity_from_time(self, molecule: int, time: float):
        intensities = dict()
        for channel in self.get_channels():
            time_arr = self.get_time(channel)
            if time_arr[0] <= time <= time_arr[-1]:
                intensities[channel] = (
                    self._data[channel]
                    .intensity.sel(molecule=molecule, time=time, method="nearest")
                    .item()
                )
        return intensities

    def to_mat(self, path: Path):
        intensities = dict()
        for channel in self.get_channels():
            intensities[channel.em] = self._data[channel].intensity.values
        sio.savemat(path, {"traces": intensities})

    @classmethod
    def from_npz_eb(cls, path: Path):
        traces = cls.from_npz(path)

        eb_file_path = path.with_name((path.stem[:-7] + "_eb.dat"))
        if not eb_file_path.exists():
            return traces
        eb_file = sio.loadmat(eb_file_path)["eb_result"]

        for channel in traces.get_channels():
            n_frames = len(traces.get_time(channel))
            vit = np.zeros((traces.n_traces, n_frames))
            for molecule in range(traces.n_traces):
                vit[molecule, :] = eb_file[channel.em][0, 0]["Vit"][0, 0][0, molecule][
                    "x"
                ].squeeze()
            array = xr.DataArray(
                vit,
                dims=["molecule", "time"],
                coords={
                    "molecule": (["molecule"], np.arange(traces.n_traces)),
                    "time": (["time"], traces.get_time(channel)),
                },
            )
            traces._data[channel] = traces._data[channel].assign(
                {"viterbi_path": array}
            )
        return traces
