import os.path
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import blink.binding_kinetics as bk
import blink.time_series as ts
import blink.photobleaching_analysis as pa
import blink.image_processing as imp
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator



criteria = 1000
last_intensity_criteria = 1000
algo_penalty = 2
algo_jump_size = 5
y_legend = 'alexa-488'
n_step1 = []
n_step2 = []
n_step3 = []
n_step4 = []
n_step_over_4 = []
n_step_lessequal_4 = []



def plot_photobleach_step(x, y, step_x, step_y, y_legend, title=None):

    """
    Plot intensity trace and step fitting curve in one figure

    Args:
        x (list): List to plot on x-axis.
        y (list): List to plot on y-axis.
        step_x (list): Change point on x coordinate.
        step_y (list): Change point on y coordinate.
        y_legend (str): Legend to show.
    """

    # Replace default parameter
    mpl.rc(
        "axes",
        linewidth=1.5,
        titlesize=18,
        titleweight="bold",
        titlepad=20,
        labelsize=14,
        labelpad=10,
        labelweight="bold",
    )
    mpl.rc("font", family="Arial")
    mpl.rc("lines", solid_capstyle="round", linewidth=1.0)
    mpl.rc("xtick", labelsize=12)
    mpl.rc("xtick.major", width=1.5, pad=5, size=5)
    mpl.rc("xtick.minor", width=1.0, size=3)
    mpl.rc("ytick", labelsize=12)
    mpl.rc("ytick.major", width=1.5, pad=5)
    mpl.rc("ytick.minor", width=1.0, size=3)
    mpl.rc("legend", frameon=False)

    # Create a figure (class) from plt.subplots along with one
    # axes (class) named ax1.
    (
        fig1,
        ax1,
    ) = plt.subplots(1, 1, figsize=(7.32, 6), sharex=True)

    # Plot xy and step_xy
    ax1.plot(x, y, color="#9D9D9C", label=y_legend, alpha=0.6)
    ax1.plot(
        step_x,
        step_y,
        color="#F94151",
        label="Fitting curve",
        drawstyle="steps-post",
        linewidth=2.0,
        alpha=0.8,
    )

    # Settings for ax1
    ax1.set_title(title)

    ax1.set_ylabel("Intensity")
    ax1.set_ylim(
        0,
    )
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.set_xlim(
        0,
    )
    ax1.set_xlabel("Time (s)")

    ax1.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.0, 1.0))


def get_change_points(data, data_criteria, algo_penalty, jump=None):
    """
    Use pelt method to detect change point in data and filter by given
    criteria. You can directly use the return two list to plot traces.

    Args:
        data (list): 1-D data for change point analysis.

        data_criteria (float or int):
            Criteria for valid change point if mean y-data before change point
            and after change point is larger than or equal to criteria.

        algo_penalty (int): Penalty for petl change point analysis.

    Returns:
        Valid change points (list): Include 0 at first index and
                                    len(data) at last index.
        Step intensity (list): Include repeated data at -1 and -2 index.
    """

    # Changepoint detection with the Pelt search method
    model = rpt.Pelt(model="rbf", jump=jump)
    change_points = model.fit_predict(signal=data, pen=algo_penalty)
    

    # Filter valid change points that match criteria.
    start_point = 0
    valid_change_points = []
    step_intensity = []
    for i, change_point in enumerate(change_points):
        change_point = int(change_point)

        # If index equal to last index
        if i == len(change_points) - 1:
            previous_point = start_point
            valid_change_points.append(len(data) - 1)
            intensity_before = np.mean(data[previous_point:change_point])
            step_intensity.append(intensity_before)
            step_intensity.append(intensity_before)

        else:
            previous_point = start_point
            next_point = change_points[i + 1]
            data_before = data[previous_point:change_point]
            data_after = data[change_point:next_point]

            # If data_before and data_after is not empty list
            if len(data_before) >= 1 and len(data_after) >= 1:

                intensity_before = np.mean(data_before)
                intensity_after = np.mean(data_after)

                if intensity_before - intensity_after >= data_criteria:
                    valid_change_points.append(change_point)
                    step_intensity.append(intensity_before)
                    start_point = change_point
                
                elif intensity_after - intensity_before >= data_criteria:
                    valid_change_points.append(change_point)
                    step_intensity.append(intensity_before)
                    start_point = change_point

    # Insert 0 at [0] position to match len(step_intensity)
    valid_change_points.insert(0, 0)

    return valid_change_points, step_intensity

def plot_and_save1 (path, category_path, save_directory):

    traces = ts.TimeTraces.from_npz(path)
    
    category = np.load(category_path)
    channel = imp.Channel("blue", "blue")
    relative_time = traces.get_time(channel)
    intensity_arr = []
    n = 0

    for i , analyzable in enumerate(category):
        if not analyzable:
            continue
        else:
            intensity = traces.get_intensity(channel, i)
            intensity_arr.append(intensity)
    intensity_arr = np.vstack(intensity_arr).T
    # print(intensity_arr.shape)

    total_aoi = len(intensity_arr.T)
    # print(len(intensity_arr.T))
    step_info = {}
    for aoi, _ in enumerate(intensity_arr):
        if aoi == 95:
            break
        intensity = intensity_arr[:, aoi]

        step_x, step_y = get_change_points(
            data=intensity,
            data_criteria=criteria,
            algo_penalty=algo_penalty,
            jump=algo_jump_size,
        )

        print(step_x)
        print(step_y)
        step_x[-1] = step_x[-1] - 1
        step_x = relative_time[step_x]

        step_count = len(step_x) - 3
        step_key = f"{step_count}-step"
        if step_count == 1:
            n_step1.append(step_count)
        if step_count == 2:
            n_step2.append(step_count)
        if step_count == 3:
            n_step3.append(step_count)
        if step_count == 4:
            n_step4.append(step_count)
        if step_count > 4:
            n_step_over_4.append(step_count)
        if step_count <= 4 and step_count!=0:
            n_step_lessequal_4.append(step_count)
        
        
        if step_y[-1] < last_intensity_criteria:

            if step_key in step_info.keys():
                step_info[step_key] += 1
            else:
                step_info[step_key] = 1

            plot_photobleach_step(
                x=relative_time,
                y=intensity,
                step_x=step_x,
                step_y=step_y,
                y_legend=y_legend,
            )

            save_name = f"dT13_binding_{n}.png"
            plt.savefig(
                os.path.join(save_directory, save_name),
                dpi=300,
                bbox_inches="tight",
                transparent=False,
            )
            # plt.show()
            n += 1
            plt.close("all")
            
    n_step_arr = [len(n_step1), len(n_step2), len(n_step3), len(n_step4), len(n_step_over_4), len(n_step_lessequal_4)]
    return n_step_arr

def plot_and_save2 (path, category_path, save_directory):

    traces = ts.TimeTraces.from_npz(path)
    
    category = np.load(category_path)
    channel = imp.Channel("blue", "blue")
    relative_time = traces.get_time(channel)
    intensity_arr = []
    n = 0

    for i , analyzable in enumerate(category):
        if not analyzable:
            continue
        else:
            intensity = traces.get_intensity(channel, i)
            intensity_arr.append(intensity)
    intensity_arr = np.vstack(intensity_arr).T
    # print(intensity_arr.shape)

    total_aoi = len(intensity_arr.T)
    # print(len(intensity_arr.T))
    step_info = {}
    for aoi, _ in enumerate(intensity_arr):
        if aoi == 117:
            break
        intensity = intensity_arr[:, aoi]

        step_x, step_y = get_change_points(
            data=intensity,
            data_criteria=criteria,
            algo_penalty=algo_penalty,
            jump=algo_jump_size,
        )

        print(step_x)
        print(step_y)
        step_x[-1] = step_x[-1] - 1
        step_x = relative_time[step_x]

        step_count = len(step_x) - 3
        step_key = f"{step_count}-step"
        if step_count == 1:
            n_step1.append(step_count)
        if step_count == 2:
            n_step2.append(step_count)
        if step_count == 3:
            n_step3.append(step_count)
        if step_count == 4:
            n_step4.append(step_count)
        if step_count > 4:
            n_step_over_4.append(step_count)
        if step_count <= 4 and step_count !=0:
            n_step_lessequal_4.append(step_count)
        
        
        if step_y[-1] < last_intensity_criteria:

            if step_key in step_info.keys():
                step_info[step_key] += 1
            else:
                step_info[step_key] = 1

            plot_photobleach_step(
                x=relative_time,
                y=intensity,
                step_x=step_x,
                step_y=step_y,
                y_legend=y_legend,
            )

            save_name = f"dT13_binding1_{n}.png"
            plt.savefig(
                os.path.join(save_directory, save_name),
                dpi=300,
                bbox_inches="tight",
                transparent=False,
            )
            # plt.show()
            n += 1
            plt.close("all")
    n_step_arr = [len(n_step1), len(n_step2), len(n_step3), len(n_step4), len(n_step_over_4), len(n_step_lessequal_4)]
    return n_step_arr


t_path1 = Path(r'D:\CWH\20221222\aoi\dT13_binding_gr_inverse_traces.npz')
c_path1 = Path(r'D:\CWH\20221222\aoi\dT13_binding_gr_inverse_category.npy')
t_path2 = Path(r'D:\CWH\20221222\aoi\dT13_binding1_gr_inverse_traces.npz')
c_path2 = Path(r'D:\CWH\20221222\aoi\dT13_binding1_gr_inverse_category.npy')
exp1 =  plot_and_save1(t_path1, c_path1, save_directory = r"D:\CWH\binding_curve_and_time_trace\dT13_binding_10nM_1")

exp2 =  plot_and_save2(t_path2, c_path2, save_directory = r"D:\CWH\binding_curve_and_time_trace\dT13_binding_10nM_2")


total_n_step = []

# add two trace into one list
for i in range(len(exp1)):
    newvalue = exp1[i] + exp2[i]
    total_n_step.append(newvalue)

print(total_n_step)

x_label= ['1', '2', '3', '4', '>4', '<= 4']

plt.bar(range(1,(len(exp1)+1)), total_n_step, color = 'darkgoldenrod', label = 'dT13 binding')

plt.xlabel('number of step')
plt.ylabel('number of event')
plt.xticks(range(1,(len(exp1)+1)),x_label)
plt.legend(fontsize = 15)
plt.show()