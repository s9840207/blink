import numpy as np
import matplotlib.pylab as plt
import ruptures as rpt
from pathlib import Path
import blink.binding_kinetics as bk
import blink.time_series as ts
import blink.photobleaching_analysis as pa
import blink.image_processing as imp

# creation of data
path=Path(r'D:\CWH\20221222\1\dT13_binding_gr_inverse_traces.npz')
traces = ts.TimeTraces.from_npz(path)
category_path = Path(r'D:\CWH\20221222\1\dT13_binding_gr_inverse_category.npy')
category = np.load(category_path)
for i in category:
    if i == 1:
        print(i)
# print(category)
channel = imp.Channel("blue", "blue")
data = []
# for i , analyzable in enumerate(category):
#     if not analyzable:
#         continue
#     else:
#         intensity = traces.get_intensity(channel, i)
#         data.append(intensity)
# data = np.vstack(data).T
# print(data.shape)



time = traces.get_time(channel)




# for aoi, _ in enumerate(df.columns):
#         intensity = intensity_arr[:, aoi]

#         step_x, step_y = tools.get_change_points(
#             data=intensity,
#             data_criteria=criteria,
#             algo_penalty=algo_penalty,
#             jump=algo_jump_size,
#         )

#         step_x[-1] = step_x[-1] - 1
#         step_x = relative_time.iloc[step_x]

#         step_count = len(step_x) - 2
#         step_key = f"{step_count}-step"

#         if step_y[-1] < last_intensity_criteria:

#             if step_key in step_info.keys():
#                 step_info[step_key] += 1
#             else:
#                 step_info[step_key] = 1

#             figures.plot_photobleach_step(
#                 x=relative_time,
#                 y=intensity,
#                 step_x=step_x,
#                 step_y=step_y,
#                 y_legend=y_legend,
#             )

#             save_name = f"{stepkey}{splitfilename[0]}{aoi}_{i}.png"
#             plt.savefig(
#                 os.path.join(save_directory, save_name),
#                 dpi=300,
#                 bbox_inches="tight",
#                 transparent=False,
#             )
#             plt.close("all")