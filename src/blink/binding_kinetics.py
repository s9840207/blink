#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (binding_kinetics.py) is part of blink.
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

from collections import namedtuple
from typing import List

import numpy as np


def collect_all_channel_state_info(traces):
    state_info_all = dict()
    StateInfo = namedtuple("StateInfo", ("n_states", "is_lowest_state_nonzero"))
    for channel in traces.get_channels():
        state_info_all[channel] = []
        for molecule in range(traces.n_traces):
            intensity = traces.get_intensity(channel, molecule)
            viterbi_path = traces.get_viterbi_path(channel, molecule)
            state_means = sorted(set(viterbi_path.data))
            is_lowest_state_nonzero = abs(state_means[0]) > 2 * np.std(
                intensity[viterbi_path == state_means[0]]
            )
            state_info_all[channel].append(
                StateInfo(len(state_means), is_lowest_state_nonzero)
            )
    return state_info_all


def list_multiple_tethers(channel_state_info: list) -> tuple:
    bad_molecules = []
    for molecule, (n_states, is_lowest_state_nonzero) in enumerate(channel_state_info):
        exists_more_than_two_states = n_states > 2
        is_two_state_but_elevated = n_states == 2 and is_lowest_state_nonzero
        if exists_more_than_two_states or is_two_state_but_elevated:
            bad_molecules.append(molecule)
    bad_molecules = tuple(bad_molecules)
    return bad_molecules


def list_none_ctl_positions(channel_state_info) -> List[int]:
    aoi_list = []
    for molecule, (n_states, is_lowest_state_nonzero) in enumerate(channel_state_info):
        if n_states != 1 or is_lowest_state_nonzero:
            aoi_list.append(molecule)
    return aoi_list


def get_interval_slices(viterbi_path):
    state_end_index = find_state_end_point(viterbi_path)
    state_start_index = [0] + (state_end_index + 1).tolist()
    state_end_index = (state_end_index + 1).tolist() + [len(viterbi_path)]
    return (
        slice(start, stop) for start, stop in zip(state_start_index, state_end_index)
    )


def colocalization_analysis(traces, state_info, binder_channel):
    bad_molecules = []
    for molecule in range(traces.n_traces):
        molecule_state_info = state_info[binder_channel][molecule]
        viterbi_path = traces.get_viterbi_path(binder_channel, molecule)
        is_colocalized = traces.get_is_colocalized(binder_channel, molecule)
        min_state_intensity = viterbi_path.min()
        interval_slices = get_interval_slices(viterbi_path)
        for interval_slice in interval_slices:
            is_zero_state = (
                viterbi_path[interval_slice.start] == min_state_intensity
                and not molecule_state_info.is_lowest_state_nonzero
            )
            if is_zero_state:
                continue

            n_colocalized_time_points = np.count_nonzero(is_colocalized[interval_slice])
            # Viterbi on state must correspond to colocalized event, with 10% tolerance
            reject_condition = n_colocalized_time_points < 0.9 * (
                interval_slice.stop - interval_slice.start
            )
            if reject_condition:
                bad_molecules.append(molecule)
                break
    return bad_molecules


def find_state_end_point(state_sequence):
    change_array = np.diff(state_sequence)
    state_end_index = np.nonzero(change_array)[0]
    return state_end_index


def assign_event_time(time_for_each_frame, state_end_index):
    event_time = np.zeros((len(state_end_index) + 2))
    event_time[0] = time_for_each_frame[0]
    event_time[-1] = time_for_each_frame[-1]
    # Assign the time point for events as the mid-point between two points that have
    # different state labels
    for i, i_end_index in enumerate(state_end_index):
        event_time[i + 1] = (
            time_for_each_frame[i_end_index] + time_for_each_frame[i_end_index + 1]
        ) / 2
    return event_time
