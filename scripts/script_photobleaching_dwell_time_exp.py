from pathlib import Path

import image_processing as imp
import numpy as np
import pandas as pd
import time_series

from blink import binding_kinetics as bk
from blink import find_two_state_dwell_time as dt


def main():
    datapath = Path(
        "/mnt/data/Research/PriA_project/analysis_result/" "20211001/20211001imscroll"
    )
    parameter_file_path = Path(
        "/mnt/data/Research/PriA_project/analysis_result/"
        "20211001/20211001parameterFile.xlsx"
    )
    sheet = "L3"
    parameters = pd.read_excel(parameter_file_path, sheet_name=sheet)
    channel = imp.Channel("blue", "blue")
    dfs = []
    for filestr in parameters.filename:

        categories = np.load(datapath / (filestr + "_category.npy"))
        traces = time_series.TimeTraces.from_npz_eb(
            datapath / (filestr + "_traces.npz")
        )
        time_array = traces.get_time(channel)

        state_info = bk.collect_all_channel_state_info(traces)
        dwells = []
        for molecule in range(traces.n_traces):
            if categories[molecule] != 1:
                continue
            i_state_info = state_info[channel][molecule]
            n_states_cat = int(
                i_state_info.n_states - 1 + i_state_info.is_lowest_state_nonzero
            )
            if n_states_cat != 1:
                continue
            state_sequence = traces.get_viterbi_path(channel, molecule)
            state_end_index = bk.find_state_end_point(state_sequence)
            event_time = bk.assign_event_time(time_array, state_end_index)
            dwells.append(event_time[1] - event_time[0])
        maxtime = time_array[-1] - time_array[0]
        dwells = np.array(dwells)
        censored = np.where(dwells == maxtime, 0, 1)
        df = pd.DataFrame(data={"time": dwells, "status": censored})
        dfs.append(df)
    df = pd.concat(dfs)
    survival_fitter = dt.SurvivalFitter(df, model="exp")
    survival_fitter.estimate_survival_curve_r()
    survival_fitter.estimate_model_parameter()
    save_fig_path = datapath / (sheet + "_pb.svg")
    survival_fitter.plot(save_fig_path)
    data_file_path = save_fig_path.with_suffix(".npz")
    survival_fitter.to_npz(data_file_path)


if __name__ == "__main__":
    main()
