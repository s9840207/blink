from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def main():
    base_dir = Path("/home/tzu-yu/Downloads/LYT/Cdc13-WT 5 nM/")
    mapping = {1: 3, 2: 2, 3: 1}
    period = 0.05
    dead_time = pd.read_csv(
        base_dir / "dead_time.csv",
        names=["date", "repeat", "dead_time"],
        dtype={"date": str, "repeat": str, "dead_time": np.float64},
        na_filter=False,
    )
    observation_table = {
        "molecule": [],
        "time": [],
        "state": [],
    }
    for _, row in dead_time.iterrows():
        if row["repeat"]:
            data_dir = base_dir / row["date"] / row["repeat"]
            filestr = row["date"] + "-" + row["repeat"]
        else:
            data_dir = base_dir / row["date"]
            filestr = row["date"]
        datapath = data_dir / "Define_state.mat"
        first_observed_index = row["dead_time"]
        state_sequences = sio.loadmat(datapath)["define_state"].squeeze()
        for i_molecule, trace in enumerate(state_sequences):
            trace = trace.squeeze()
            for i_frame, state in enumerate(trace):
                if i_frame == 0:
                    observation_table["molecule"].append(
                        (filestr + "-" + str(i_molecule))
                    )
                    observation_table["time"].append(i_frame * period)
                    observation_table["state"].append(1)
                elif i_frame >= first_observed_index:
                    observation_table["molecule"].append(
                        (filestr + "-" + str(i_molecule))
                    )
                    observation_table["time"].append(i_frame * period)
                    observation_table["state"].append(mapping[state])
    pd.DataFrame(observation_table).to_csv((base_dir / "observations.csv"), index=False)


if __name__ == "__main__":
    main()
