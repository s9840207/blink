#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (find_two_state_dwell_time.py) is part of blink.
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

"""This is a temporary script that analyzes the dwell times of two state model."""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri
import seaborn as sns
import strictyaml
import xarray as xr
from matplotlib import pyplot as plt
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from scipy import optimize

import gui.file_selectors
from blink import binding_kinetics, utils

rpy2.robjects.pandas2ri.activate()
r_survival = importr("survival")


def make_first_dwell_survival_time_df(intervals: "Intervals"):
    n_right_censored = intervals.count_flat_traces()
    dwells = intervals.get_two_state_first_event_time()
    interval_censor_table = np.zeros((len(dwells.duration) + n_right_censored, 2))

    interval_censor_table[0 : len(dwells.duration), 0] = dwells.duration.values
    interval_censor_table[0 : len(dwells.duration), 1] = xr.where(
        dwells.event_observed, 1, 2
    ).values
    interval_censor_table[len(dwells.duration) :, 0] = intervals.get_max_time()
    interval_censor_table[len(dwells.duration) :, 1] = 0
    df = pd.DataFrame(interval_censor_table, columns=["time", "status"])
    return df


def make_first_high_survival_time_df(intervals: "Intervals"):
    n_right_censored = intervals.count_flat_traces()
    dwells = intervals.get_two_state_first_event_time(state="high")
    interval_censor_table = np.zeros((len(dwells.duration) + n_right_censored, 2))

    interval_censor_table[0 : len(dwells.duration), 0] = dwells.duration.values
    interval_censor_table[0 : len(dwells.duration), 1] = xr.where(
        dwells.event_observed, 1, 2
    ).values
    interval_censor_table[len(dwells.duration) :, 0] = intervals.get_max_time()
    interval_censor_table[len(dwells.duration) :, 1] = 0
    df = pd.DataFrame(interval_censor_table, columns=["time", "status"])
    return df


def make_high_dwell_survival_time_df(intervals: "Intervals"):
    dwells = intervals.get_two_state_dwell_time(state=1)
    dwells_df = dwells.to_dataframe()[["duration", "event_observed"]]
    dwells_df = dwells_df.rename(
        columns={"duration": "time", "event_observed": "status"}
    )
    dwells_df["status"] = dwells_df["status"].astype(int)
    return dwells_df


def calculate_on_off_rate(intervals, save_path):
    df = make_first_dwell_survival_time_df(intervals)
    survival_fitter = SurvivalFitter(df, model="bi-exp")
    survival_fitter.estimate_survival_curve_r()
    survival_fitter.estimate_model_parameter()
    save_fig_path = save_path.with_name(save_path.name + "_first_on.svg")
    survival_fitter.plot(save_fig_path)
    data_file_path = save_fig_path.with_suffix(".npz")
    survival_fitter.to_npz(data_file_path)

    df = make_high_dwell_survival_time_df(intervals)
    survival_fitter = SurvivalFitter(df, model="exp")
    survival_fitter.estimate_survival_curve_r()
    survival_fitter.estimate_model_parameter()
    save_fig_path = save_path.with_name(save_path.name + "_off.svg")
    survival_fitter.plot(save_fig_path)
    data_file_path = save_fig_path.with_suffix(".npz")
    survival_fitter.to_npz(data_file_path)


def read_excluded_aois(filestr: str, datapath: Path):
    discarded_txt_path = datapath.with_name(datapath.parent.name + "_discarded.txt")
    if discarded_txt_path.is_file():
        with open(discarded_txt_path, "r") as f:
            content = f.read()
        excluded_aois_dict = strictyaml.load(content)
        excluded_aois = [
            int(i) for i in excluded_aois_dict[filestr].data.strip(",").split(",")
        ]
        return excluded_aois
    else:
        raise FileNotFoundError(f"File {str(discarded_txt_path)} is not found.")


def find_on_off_rate(filestr: str, datapath: Path, parameter_df):
    all_file_path = datapath / (filestr + "_all.json")
    interval_object = Intervals.from_all_file(all_file_path)
    interval_object.exclude_aois(read_excluded_aois(filestr, datapath))
    time_offset = parameter_df.loc[parameter_df["filename"] == filestr, "time offset"]
    interval_object.set_time_offset(time_offset)
    save_path = all_file_path.with_name(filestr)
    calculate_on_off_rate(interval_object, save_path)


def log_sum_exp(arr):
    x_max = arr.max(axis=0)
    result = x_max + np.log(np.sum(np.exp(arr - x_max), axis=0))
    return result


def sum_log_f(log_t, log_k1, log_k2, log_A):
    term1 = log_A + log_k1 - np.exp(log_k1 + log_t)
    term2 = log1mexp(-log_A) + log_k2 - np.exp(log_k2 + log_t)
    log_f_arr = log_sum_exp(np.stack((term1, term2), axis=0))
    return log_f_arr.sum()


def log_S(log_t, log_k1, log_k2, log_A):
    term1 = log_A - np.exp(log_k1 + log_t)
    term2 = log1mexp(-log_A) - np.exp(log_k2 + log_t)
    log_S_arr = log_sum_exp(np.stack((term1, term2), axis=0))
    return log_S_arr


def log1mexp(x):
    if x > np.log(2):
        result = np.log1p(-np.exp(-x))
    else:
        result = np.log(-np.expm1(-x))
    return result


def fit_biexponential(data):
    def n_log_lik(log_param):
        observed = data.time[data.status == 1].to_numpy()
        right_censored = data.time[data.status == 0].to_numpy()
        left_censored = data.time[data.status == 2].to_numpy()
        return -(
            sum_log_f(np.log(observed), *log_param)
            + np.sum(log_S(np.log(right_censored), *log_param))
            + np.sum(np.log(1 - np.exp(log_S(np.log(left_censored), *log_param))))
        )

    k_guess = 1 / np.mean(data.time)
    result = optimize.minimize(
        n_log_lik,
        [np.log(k_guess), np.log(k_guess / 1.5), np.log(0.5)],
        bounds=((np.log(1e-7), 0), (np.log(1e-7), 0), (-100, -1e-16)),
        method="L-BFGS-B",
    )
    return result


class SurvivalFitter:
    def __init__(self, survival_data: pd.DataFrame, model: str):
        self.data = survival_data
        self.model = model
        self.param = None

    def estimate_survival_curve_r(self):
        with localconverter(
            robjects.default_converter + rpy2.robjects.pandas2ri.converter
        ):
            r_from_pd_df = robjects.conversion.py2rpy(self.data)
        robjects.r(
            f"surv <- with({r_from_pd_df.r_repr()}, "
            f'Surv(time=time, time2=time, event=status, type="interval"))'
        )
        robjects.r("fit <- survfit(surv~1, data={})".format(r_from_pd_df.r_repr()))
        robjects.r("fit0 <- survfit0(fit)")
        output = dict()
        output["time"] = robjects.r('fit0[["time"]]')
        output["surv"] = robjects.r('fit0[["surv"]]')
        output["upper_ci"] = robjects.r('fit0[["upper"]]')
        output["lower_ci"] = robjects.r('fit0[["lower"]]')
        self.survival_curve = pd.DataFrame(
            output, columns=["time", "surv", "upper_ci", "lower_ci"]
        )

    def estimate_model_parameter(self):
        if self.model == "exp":
            self._model_func = lambda t, k: np.exp(-k * t)
            with localconverter(
                robjects.default_converter + rpy2.robjects.pandas2ri.converter
            ):
                r_from_pd_df = robjects.conversion.py2rpy(self.data)
            robjects.r(
                f"surv <- with({r_from_pd_df.r_repr()}, "
                f'Surv(time=time, time2=time, event=status, type="interval"))'
            )
            robjects.r(
                'exreg <- survreg(surv~1, data={}, dist="exponential")'.format(
                    r_from_pd_df.r_repr()
                )
            )
            intercept = robjects.r('exreg["coefficients"]')[0].item()
            # log_var = robjects.r('exreg["var"]')[0].item()
            self.param = [np.exp(-intercept)]
        elif self.model == "bi-exp":
            self._model_func = lambda t, k1, k2, A: A * np.exp(-k1 * t) + (
                1 - A
            ) * np.exp(-k2 * t)
            result = fit_biexponential(self.data)
            self.param = np.exp(result.x)

    def plot(self, save_fig_path: Path = None):
        color = sns.color_palette(palette="muted")[0]
        fill_color = sns.color_palette(palette="pastel")[0] + (80 / 255,)
        surv = self.survival_curve
        x = np.linspace(
            0, surv["time"].iloc[-1], int(round(surv["time"].iloc[-1] * 10))
        )
        y = self._model_func(x, *self.param)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.despine(fig, ax)
        ax.plot(x, y, color=sns.color_palette(palette="muted")[1])
        ax.step(surv["time"], surv["surv"], where="post", color=color)
        ax.fill_between(
            surv["time"],
            surv["lower_ci"],
            surv["upper_ci"],
            step="post",
            color=fill_color,
        )
        n_total = len(self.data["status"])
        n_right_censored = n_total - np.count_nonzero(self.data["status"])
        n_left_censored = np.count_nonzero(self.data["status"] == 2)
        stat_counts = (n_total, n_right_censored, n_left_censored)
        string = "{}, r {}, l {}".format(*stat_counts)
        ax.text(0.6, 0.6, string, transform=ax.transAxes, fontsize=14)

        ax.set_ylim((0, 1.05))
        ax.set_xlim((0, surv["time"].iloc[-1]))
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Survival probability", fontsize=14)
        plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(save_fig_path, format="svg", dpi=300, bbox_inches="tight")
        plt.close()

    def to_hdf5(self, path: Path):
        with h5py.File(path, "w") as f:
            column_keys = np.array(
                ["time", "survival", "upper_ci", "lower_ci"], dtype="S"
            )
            group_survival_curve = f.create_group("survival_curve")
            group_survival_curve.create_dataset(
                "column_keys", column_keys.shape, column_keys.dtype, column_keys
            )
            surv_data = self.survival_curve.to_numpy()
            group_survival_curve.create_dataset("data", surv_data.shape, "f", surv_data)
            group_exp_model = f.create_group("bi_exp_model")
            group_exp_model.create_dataset("param", (3,), "f", self.param)

    def to_npz(self, path: Path):
        np.savez(
            path,
            data=self.data.to_records(),
            survival_curve=self.survival_curve.to_records(),
            model=np.array([self.model], dtype="U10"),
            param=self.param,
        )


def read_time_offset(parameter_file_path: Path, sheet: str):
    dfs = utils.read_excel(parameter_file_path, sheet_name=sheet)
    return dfs["time offset"][0]


class Intervals:
    def __init__(self):
        self.data = None
        self.aoi_category = dict()
        self.max_time = 0
        self.excluded_aois = tuple()
        self.time_offset = 0

    @classmethod
    def from_all_file(cls, all_file_path: Path) -> "Intervals":
        intervals = cls()
        all_data, intervals.aoi_category = binding_kinetics.load_all_data(all_file_path)
        intervals.data = all_data["intervals"]
        intervals.max_time = all_data["data"].time.values.max()
        return intervals

    def exclude_aois(self, aoi_numbers: Tuple[int]):
        self.excluded_aois = aoi_numbers

    def count(self, option: str) -> int:
        pass

    def _get_aoi_idx(self, category: str) -> Tuple[int]:
        idx_list = [
            i
            for i in self.aoi_category["analyzable"][category]
            if i not in self.excluded_aois
        ]
        return tuple(idx_list)

    def count_flat_traces(self, category="0"):
        return len(self._get_aoi_idx(category=category))

    def get_two_state_first_event_time(self, state="low"):
        if state == "low":
            two_state_aoi = self._get_aoi_idx(category="1")
            first_intervals = self.data.sel(interval_number=0, AOI=list(two_state_aoi))
            left_censored = first_intervals.state_number == 1
        elif state == "high":
            two_state_aoi = self._get_aoi_idx(category="0")
            first_intervals = self.data.sel(interval_number=0, AOI=list(two_state_aoi))
            left_censored = first_intervals.state_number == -1
        dwells = first_intervals["duration"].to_dataset()
        dwells["duration"] = (
            xr.where(left_censored, 0, dwells["duration"]) + self.time_offset
        )
        dwells["event_observed"] = np.logical_not(left_censored)
        return dwells

    def set_time_offset(self, value):
        self.time_offset = float(value)

    def get_max_time(self):
        return self.max_time + self.time_offset

    def get_two_state_dwell_time(self, state: int):
        out_list = []
        for iaoi in self.aoi_category["analyzable"]["1"]:
            if iaoi not in self.excluded_aois:
                i_aoi_intervals = self.data.sel(AOI=iaoi)
                valid_intervals = i_aoi_intervals.where(
                    np.logical_not(np.isnan(i_aoi_intervals.duration)), drop=True
                )

                if len(valid_intervals.duration) != 0:
                    valid_intervals = valid_intervals.assign(
                        {
                            "event_observed": (
                                "interval_number",
                                np.ones(len(valid_intervals.interval_number)),
                            )
                        }
                    )

                    valid_intervals["event_observed"][[0, -1]] = 0
                    i_dwell = valid_intervals[["duration", "event_observed"]].where(
                        valid_intervals.state_number == state, drop=True
                    )

                    i_dwell["event_observed"] = i_dwell["event_observed"].astype(bool)
                    i_dwell = i_dwell.reset_index("interval_number")

                    out_list.append(i_dwell)

        out = xr.concat(out_list, dim="interval_number")

        return out


def main():
    """main function"""
    xlsx_parameter_file_path = gui.file_selectors.get_xlsx_parameter_file_path()
    datapath = gui.file_selectors.def_data_path()
    dfs = utils.read_excel(xlsx_parameter_file_path, sheet_name="all")
    for filestr in dfs["filename"]:
        try:
            find_on_off_rate(filestr, datapath, dfs)
        except ValueError as e:
            print(repr(e))
            continue


if __name__ == "__main__":
    main()
