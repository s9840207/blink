import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager



label_font  = {'family':'arial','color':'black','size':10, 'weight':'bold'}
title_font = {'family':'arial','color':'black','size':18, 'weight':'bold'}
color = [ '#9D9D9D', '#984B4B', '#7373B9' ]
label = ['FERT = 0.5', 'FRET = 0.6', 'FRET = 0.8']


path =Path(r'D:\CWH\20230523/123.xlsx')
data=pd.read_excel(path)
print(data)
fret_raw_data = data.iloc[:,1:6]
fret = np.array(fret_raw_data.values.tolist())
# x=a.values.tolist()
# y=b.values.tolist()
# xdata=np.array(x)
# ydata=np.array(y)
print(fret)
x = np.array([0, 62.5, 125, 250, 500])
print(x)
print(fret[0,0:4])

fig, ax = plt.subplots()
for i in range(len(fret[:,0])):
    ax.plot(x, fret[i,0:5], color = color[i])
    ax.scatter(x, fret[i,0:5], color = color[i], label = label[i])
    ax.legend()

plt.xlabel('[BCDX2](nM)', fontdict = label_font)
plt.ylabel('Fraction', fontdict = label_font)
plt.xlim(-10,550)
plt.ylim(0,0.70)
plt.title('FRET scattering', fontdict = title_font)
plt.savefig(Path(r'D:\CWH\20230523\fret_scatter', dpi = 1200), bbox_inches="tight")
plt.show()