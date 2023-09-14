import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import math
import matplotlib.font_manager as font_manager
from pathlib import Path
# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False
 
#Fitting function
def func(x, n, k):
    return (x**n)/(k**n + x**n)
    

# def uncertainty(a, b) :
#     n = ufloat(a[0], b[0])
#     kd = ufloat(a[1], b[1])
#     k = kd**(1/n)
#     return k
    
data_color = [ '#9D9D9D', '#BC8F8F' ]
label = ['ADP', 'ATP']
font = font_manager.FontProperties(family='arial',
                                   weight='bold',
                                   style='normal', size=10)
label_font  = {'family':'arial','color':'black','size':18, 'weight':'bold'}
title_font = {'family':'arial','color':'black','size':12, 'weight':'bold'}
data_folder = Path(r'D:\CWH\2023\20230719')
filestr = 'bound_fraction'
data_path = (data_folder / (filestr + ".xlsx"))

data=pd.read_excel(data_path)
print(data)
bound_fraction = []
conc = data.iloc[:,0].values.tolist()
adp =data.iloc[0:6,1]
atp =data.iloc[0:6,6]
adp=adp.values.tolist()
atp=atp.values.tolist()
bound_fraction.append(adp)
bound_fraction.append(atp)

print(bound_fraction, conc)


# Plot experimental data points



 
 
# # Perform the curve-fit
xFit = np.arange(0, 1000, 0.01)
for i , data in enumerate(bound_fraction):
    plt.plot(conc, data, 'bo',label= label[i], color = data_color[i])
    popt, pcov = curve_fit(func, conc, data )
    perr = np.sqrt(np.diag(pcov))
    plt.plot(xFit, func(xFit, *popt),
            label='n=%5.3f, Kd= %5.3f' % tuple(popt),
            color= data_color[i],
            lw = 3
            )
    print(popt)
    print(perr)
    plt.legend(loc=4, fontsize=10, prop = font)
# k_half = popt_adp[1]**(1/popt_adp[0])

plt.xlim(-10, 1100)
plt.ylim(-0.1, 1.1)
plt.ylabel('Bound fraction',fontdict = label_font)
plt.xlabel('[RAD51] nM', fontdict = label_font)
plt.savefig(data_folder / f'{filestr}.png',
            dpi = 1200,
            bbox_inches = 'tight'
            )
plt.show()



# plt.savefig(r'D:\CWH\binding_curve_and_time_trace\binding_curve\dT27_ATP_Ca2+', dpi = 1200, bbox_inches = 'tight')
# plt.show()


