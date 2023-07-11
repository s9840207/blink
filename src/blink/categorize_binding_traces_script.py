#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (categorize_binding_traces_script.py) is part of blink.
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

"""Categorize_binding_traces_script """

import json
from pathlib import Path
from typing import List

import numpy as np

import blink.image_processing as imp
from blink import binding_kinetics as bk
from blink import time_series, utils


def categorize_binding_traces(
    parameter_file_path: Path,
    sheet_list: List[str],
    datapath: Path,
    save_file=True,
):
    """Categorize traces into bad and analyzable categories.

    Run through each entry in the specified sheets in the parameter file.
    Args:
        parameter_file_path: The path of the xlsx parameter file
        sheet_list: A list of sheet names to be analyzed.
    """
    dfs = utils.read_excel(parameter_file_path, sheet_name="channels")
    target_channel_str = dfs.loc[dfs["map file name"].isna(), "name"].to_numpy().item()
    target_channel = imp.Channel(target_channel_str, target_channel_str)
    binder_channel_list = dfs.loc[
        np.logical_not(dfs["map file name"].isna()), "name"
    ].to_list()
    if len(binder_channel_list) > 1:
        raise NotImplementedError(
            "Analysis for more than one channel is not implemented"
        )
    binder_channel = imp.Channel(binder_channel_list[0], binder_channel_list[0])
    for i_sheet in sheet_list:
        dfs = utils.read_excel(parameter_file_path, sheet_name=i_sheet)
        for filestr in dfs.filename:
            is_ctl_file = filestr[-3:] == "ctl"
            aoi_categories = {}

            try:
                data = time_series.TimeTraces.from_npz(
                    datapath / (filestr + "_traces.npz")
                )
            except FileNotFoundError:
                continue

            state_info = bk.collect_all_channel_state_info(data)
            if is_ctl_file:
                raise NotImplementedError(
                    "The control aoi option has not been rewritten."
                )
                bad_tethers = bk.list_none_ctl_positions(
                    state_info.sel(channel=data.target_channel)
                )
                aoi_categories["tethers"] = bad_tethers
            else:
                bad_tethers = bk.list_multiple_tethers(state_info[target_channel])
                aoi_categories["multiple_tethers"] = bad_tethers
            non_colocalized_aois = bk.colocalization_analysis(
                data, state_info, binder_channel
            )
            aoi_categories["false_binding"] = sorted(
                set(non_colocalized_aois) - set(bad_tethers)
            )
            analyzable_aois = sorted(
                set(range(data.n_traces)) - set(non_colocalized_aois) - set(bad_tethers)
            )
            aoi_categories["analyzable"] = group_analyzable_aois_into_state_number(
                analyzable_aois, state_info[binder_channel]
            )
            if save_file:
                save_categories(
                    aoi_categories, datapath / (filestr + "_categories.json")
                )

            print(filestr + " finished")
    return aoi_categories


def save_categories(aoi_categories, path):
    with open(path, "w") as fobj:
        json.dump({"aoi_categories": aoi_categories}, fobj)


def group_analyzable_aois_into_state_number(analyzable_aois, channel_state_info):
    analyzable_tethers = {}
    for molecule in analyzable_aois:
        state_info = channel_state_info[molecule]
        category = int(state_info.n_states - 1 + state_info.is_lowest_state_nonzero)
        if category in analyzable_tethers:
            analyzable_tethers[category].append(molecule)
        else:
            analyzable_tethers[category] = [molecule]
    return analyzable_tethers
