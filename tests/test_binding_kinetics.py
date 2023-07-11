import numpy as np
import xarray as xr

from blink import binding_kinetics as bk


def test_find_state_end_equal_steps():
    sequence = xr.DataArray(np.repeat(np.arange(5), 10), dims="time")
    state_end_index = bk.find_state_end_point(sequence)
    np.testing.assert_array_equal(state_end_index, np.array(range(9, 40, 10)))


def test_assign_event_time_equal_steps():
    sequence = xr.DataArray(
        np.repeat(np.arange(5), 10), dims="time", coords={"time": range(50)}
    )
    state_end_index = np.array(range(9, 40, 10))
    event_time = bk.assign_event_time(sequence.time.values, state_end_index)
    expected_result = np.concatenate(([0], state_end_index + 0.5, [49]))
    np.testing.assert_array_equal(event_time, expected_result)


# def test_remove_two_state_with_lowest_not_equal_to_zero(self):
#     STATES = np.repeat([1, 2, 1, 2, 1], 10)
#     AOI_1_bad_intensity = np.repeat([1, 2, 1, 2, 1], 10)
#     AOI_2_good_intensity = np.repeat([0, 1, 0, 1, 0], 10)
#     AOI_3_flat = np.ones(50)
#     AOI_1_viterbi_path = np.stack((STATES, AOI_1_bad_intensity), 0)
#     AOI_2_viterbi_path = np.stack((STATES, AOI_2_good_intensity), 0)
#     AOI_3_viterbi_path = np.ones((2, 50))
#     viterbi_path = np.stack((AOI_1_viterbi_path, AOI_2_viterbi_path, AOI_3_viterbi_path), 0)
#     intensity = np.stack((AOI_1_bad_intensity, AOI_2_good_intensity, AOI_3_flat), 0)
#     channel_data = xr.Dataset({'intensity': (['AOI', 'time'], intensity),
#                                 'viterbi_path': (['AOI', 'state', 'time'], viterbi_path)},
#                                 coords={'AOI': [1, 2, 3],
#                                         'state': ['label', 'position'],
#                                         'time': range(0, 50)})
#     channel_state_info = bk.collect_channel_state_info(channel_data)
#     bad_aoi = bk.list_aois_elevated_two_states(channel_state_info)
#     self.assertEqual({1}, set(bad_aoi))
