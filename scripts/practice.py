import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from matplotlib import rc
from pathlib import Path
import matplotlib.font_manager as font_manager

# # rc('font', **{'family':'serif','serif':['Palatino']})
# # rc('text', usetex=True)









# data_path = Path(r'D:\CWH\20230303')
# data=pd.read_excel(r"D:\CWH\20230303\20230303.xlsx")
# font = font_manager.FontProperties(family='arial',
#                                    weight='bold',
#                                    style='normal', size=12)
# label_font  = {'family':'arial','color':'black','size':10, 'weight':'bold'}
# title_font = {'family':'arial','color':'black','size':12, 'weight':'bold'}

# print(data)



# with_Ca = np.array(data.iloc[20:22,1])
# without_Ca = np.array(data.iloc[22:24,1])
# x_label = np.array([r'with $Ca^{2+}$', r'without $Ca^{2+}$'])

# # std = data.iloc[18:20,3]
# # print(x_label, bd_no_RPA, std)

# # std = np.array(std)
# # x_label = np.array(x_label)
# # x_label_RPA = x_label[0:2]
# # print(x_label)

# # print(x_label_RPA)
# plt.bar(range(len(with_Ca)), with_Ca, width = 0.3, tick_label = x_label, color = ['#5CADAD', '#5CADAD'])
# for a,b in zip(range(len(with_Ca)), with_Ca):
#     b = round(b, 3)
#     plt.text(a, b, b, ha = 'center', va = 'bottom', font = font)

# plt.bar(range(len(without_Ca)), without_Ca, width = 0.3, tick_label = x_label, color = ['#A5A552','#A5A552'])
# for a,b in zip(range(len(without_Ca)), without_Ca):
#     b = round(b, 3)
#     plt.text(a, b, b, ha = 'left', va = 'bottom', font = font)

# # plt.errorbar(range(0,2), bd_RPA[0:2], yerr = std, fmt = '_', color = 'black', elinewidth = 2 )
# plt.title('Bound fraction of RAD51 with RPA coated dT12+24', title_font)
# # plt.xlabel('[BCDX2]', label_font)
# plt.ylabel('Bound fraction', label_font)
# plt.legend(['800 nM RAD51 only', '800nM RAD51 and BCDX2'], prop = font, loc = 9)
# plt.savefig(Path(data_path / 'RAD51_bd w or wo Ca and BCDX2 of dT12+24'), dpi = 1200, bbox_inches="tight")
# plt.show()
# plt.close()


# plt.bar(range(0,2), bd_RPA, width = 0.3, tick_label = x_label_RPA, color = ['#A5A552','#A5A552'])
# for a,b in zip(range(len(bd_RPA)), bd_RPA):
#     b = round(b, 3)
#     plt.text(a, b, b, ha = 'left', va = 'bottom', font = font, color = '#A5A552')

# # plt.errorbar(range(0,2), bd_RPA[0:2], yerr = std, fmt = '_', color = 'black', elinewidth = 2)
# plt.title('Bound fraction of RAD51 with dT21', title_font)
# plt.xlabel('[BCDX2]', label_font)
# plt.ylabel('Bound fraction', label_font)
# plt.legend(['with RPA coated dT21'], prop = font, loc = 9)

# # plt.savefig(Path(data_path / 'RAD51_bd w or wo BCDX2 with RPA coated dT21'), dpi = 1200, bbox_inches="tight")
# plt.show()
# plt.close() 
# x = np.concatenate((np.random.normal(5, 5, 1000), np.random.normal(10, 2, 1000)))

# print(x)

data = {[1, 2, 3], [4, 5, 6]}
df = pd.DataFrame(data)
print(df)

