#open the Windows terminal
#change the work path 
# key in: cd D:\
# key in: .\venv\Scripts\Activate.ps1 
# venv=virtual enviroment, that is to distinguish the python program from Windows system and the code we made

# open the spot-circling program
# key in: blink view-image
# choose the want-circle file of experiment
# the file after circling should be saved in a paricular path: D:\CWH\date\circling_folder\circling_file
# note that the exp result should be saved in a paricular pathe as well: D:\CWH\date\exp_condition\gilmpse_file

# write the three-channel mapping file
# key in(in windows terminal): blink make-mapping-file D:\CWH\date\mapping_folder (吐出一個mapping_file, 裡面有三個array,分別對到三個channel的mapping)
# key in(in windows terminal): python .\blink\scripts\extract_one_chan_mapping.py

# use combine_spot.py to write the combined file 
# open the blue channel file of DNA , pick AOI, record the numbers(numbers of protein)
# load grenn AOI and recod the AOI numbers, distance threshold should be 1.5 now
# load the mapping file by loadmapping buttom 
# map the blue and green channel by inverse map buttom (map buttom is map the blue channel(short wavelength) to green channel(long wavelength))

# plot time_trace
# use python intensity_trace.py
# AOI_folder -> mapping_file_folder -> 日期 ->  excel
# excel: filename:xxx_aoi的xxx flodername:glimpse的上上層 工作表的chaneel的map file name是mapping檔的名子
# use pyhthon plot_traces
# excel -> AOI的folder

#aoi image
# click file including hwligroup -> map.npz




import pandas as pd
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


# def main (path, category_path, save_directory):

#     traces = ts.TimeTraces.from_npz(path)
    
#     category = np.load(category_path)
#     channel = imp.Channel("blue", "blue")
#     relative_time = traces.get_time(channel)
#     intensity_arr = []
#     n = 0

#     for i , analyzable in enumerate(category):
#         if not analyzable:
#             continue
#         else:
#             intensity = traces.get_intensity(channel, i)
#             intensity_arr.append(intensity)
#     intensity_arr = np.vstack(intensity_arr).T
#     # print(intensity_arr.shape)

#     print(len(intensity_arr.T))

# main(Path(r'D:\CWH\20221222\aoi\dT13_binding_gr_inverse_traces.npz'),
# Path(r'D:\CWH\20221222\aoi\dT13_binding_gr_inverse_category.npy'), save_directory = r'D:')
# main(Path(r'D:\CWH\20221222\aoi\dT13_binding1_gr_inverse_traces.npz'),
# Path(r'D:\CWH\20221222\aoi\dT13_binding1_gr_inverse_category.npy'), save_directory = r'D:')

a = np.array([1, 2, 3, 4])
print(a,np.var(a))


