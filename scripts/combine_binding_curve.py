import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.font_manager as font_manager

import math
from pathlib import Path
import os
import glob

#Fitting function
def func(x, n, kd):
    return x**n/(kd**n + x**n)
    #return a*x+b

# def uncertainty(a, b) :
#     n = ufloat(a[0], b[0])
#     kd = ufloat(a[1], b[1])
#     k = kd**(1/n)
#     return k


save_directory = Path(r'D:\CWH\excel')
data_name=[]
data_color = [ '#9D9D9D', '#B15BFF' ]#['#01B468', ]  '#B15BFF', , '#B15BFF' ['#FF60AF', '#FF0000',
data_name = [r'ATP + Ca$^{2+}$ 150mM KCl', r'ATP +Ca$^{2+}$ with 4 times BCDX2' ]#[r'ATP', r'AMP-PMP', r'ATP+Ca$^{2+}$ 50mM KCl']
font = font_manager.FontProperties(family='arial',
                                   weight='bold',
                                   style='normal', size=10)
label_font  = {'family':'arial','color':'black','size':10, 'weight':'bold'}
title_font = {'family':'arial','color':'black','size':12, 'weight':'bold'}
marker = ['v', 'v']

for i, path in enumerate(save_directory.iterdir()):
    if path.suffix != '.xlsx':
        continue
    print(i,path.stem)
    data_name.append(path.stem)
    
    


    data=pd.read_excel(path)
    print(data)
    a=data.iloc[25:31,0]
    b=data.iloc[25:31,1]
    x=a.values.tolist()
    y=b.values.tolist()
    xdata=np.array(x)
    ydata=np.array(y)

    
    # Plot experimental data points
    plt.plot(xdata, ydata, color = data_color[i], marker = marker[i], linestyle = "", label = data_name[i])

    
    
    #Perform the curve-fit
    popt, pcov = curve_fit(func, xdata, ydata, bounds=(-10,[10,1000000]))
    perr = np.sqrt(np.diag(pcov))
    print(popt)
    print(perr)

    # k_half = popt[1]**(1/popt[0])
    # print(k_half)



    # x values for the fitted function
    xFit = np.arange(0, 600, 0.01)
    
    #Plot the fitted function
    plt.plot(xFit, func(xFit, *popt), label='n= %5.3f,Kd=%5.3f' % tuple(popt), color =data_color[i])
    
    plt.xlabel('[hRAD51] (nM)', fontdict = label_font)
    plt.ylabel('Bound fraction', fontdict = label_font)
    plt.title('Bound fraction of RAD51 with dT27', title_font )
    plt.legend(loc=4, fontsize=10, prop = font)
    plt.ylim(-0.01,1)
    plt.xlim(-1,600)
    plt.savefig(Path( save_directory / data_name[i], dpi = 1200), bbox_inches="tight")
plt.show()



