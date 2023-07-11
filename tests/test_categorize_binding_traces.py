#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (test_categorize_binding_traces.py) is part of blink.
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

import json
from pathlib import Path

import xarray

import blink.image_processing as imp
from blink import binding_kinetics
from blink import categorize_binding_traces_script as cbt
from blink import time_series

TEST_DATA_DIR = Path(__file__).parent / "test_data/20200228/"


def test_get_state_info():
    true_state_info = xarray.load_dataset(TEST_DATA_DIR / "L2_state_info.netcdf")
    traces = time_series.TimeTraces.from_npz(TEST_DATA_DIR / "L2_traces.npz")
    state_info = binding_kinetics.collect_all_channel_state_info(traces)
    for channel in set(true_state_info.channel.data):
        for aoi in true_state_info.AOI:
            true_n_states = true_state_info.sel(channel=channel, AOI=aoi).nStates
            true_is_lowest_state_equal_to_zero = true_state_info.sel(
                channel=channel, AOI=aoi
            ).bool_lowest_state_equal_to_zero
            channel_obj = imp.Channel(channel, channel)
            assert state_info[channel_obj][int(aoi) - 1].n_states == true_n_states
            assert (
                not state_info[channel_obj][int(aoi) - 1].is_lowest_state_nonzero
            ) == true_is_lowest_state_equal_to_zero


def test_categorization():
    test_data_path = Path(__file__).parent / "test_data/20200228/"
    test_parameter_file_path = test_data_path / "20200228parameterFile.xlsx"

    aoi_categories = cbt.categorize_binding_traces(
        test_parameter_file_path, ["L2"], test_data_path, save_file=False
    )
    with open(test_data_path / "L2_categories.json") as f:
        true_aoi_categories = json.load(f)["aoi_categories"]

    for key in aoi_categories["analyzable"]:
        aoi_categories["analyzable"][str(key)] = list(
            aoi_categories["analyzable"].pop(key)
        )

    for k, v in aoi_categories.items():
        if k == "analyzable":
            continue
        aoi_categories[k] = list(v)

    assert aoi_categories == true_aoi_categories
