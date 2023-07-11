import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
 
#Fitting function
def func(A, k, t):
    return A * np.exp(-k * t) 
    

# def uncertainty(a, b) :
#     n = ufloat(a[0], b[0])
#     kd = ufloat(a[1], b[1])
#     k = kd**(1/n)
#     return k
    

data=pd.read_excel(r"D:\CWH\20230314\1_DNA 1uM RAD51 wash 30min_aoi\colocalization.xlsx")
print(data)
time=data.iloc[12:23,0]
col=data.iloc[12:23,1]
x=time.values.tolist()
y=col.values.tolist()
xdata=np.array(x)
ydata=np.array(y)
print(xdata, ydata)

# Plot experimental data points
plt.plot(xdata, ydata, 'bo', label=r'dT27 with ATP+$Ca^{2+}$')

 
 
#Perform the curve-fit
popt, pcov = curve_fit(func, xdata, ydata, bounds=(-10,[10000, 10000]))
perr = np.sqrt(np.diag(pcov))
print(popt)
print(perr)

# k_half = popt[1]**(1/popt[0])
# print(k_half)



# x values for the fitted function
xFit = np.arange(0, 35, 0.01)
 
#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), label='n=%5.3f, Kd= %5.3f' % tuple(popt), color='r')
 
plt.xlabel('[hRAD51] (nM)')
plt.ylabel('Bound fraction')
plt.legend(loc=2)
plt.ylim(0,1)
plt.xlim(0,35)
# plt.savefig(r'D:\CWH\binding_curve_and_time_trace\binding_curve\dT27_ATP_Ca2+', dpi = 1200, bbox_inches = 'tight')
plt.show()