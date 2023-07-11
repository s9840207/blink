import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from pathlib import Path
from scipy.stats import norm
import sklearn
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mpl
import blink.time_series as ts
import blink.image_processing as imp
from collections import Counter

def plot_his(x_axis, mean, covs, weights, fret_array, 
           color, datapath, filestr, hist_color):
  for i in range(len(mean)):
    y_axis = norm.pdf(x_axis, float(mean[i][0]),
                      np.sqrt(float(covs[i][0][0])))*weights[i]
    plt.plot(x_axis, y_axis, color = color[i], lw=3)
    plt.text(mean[i][0], max(y_axis),
            f"{mean[i][0]:.2f}: {weights[i]:.1%}",
            color=color[i],
            fontsize=15, 
            ha = 'center',
            zorder = 10
            )
  bin = np.arange(0, 1.03, 0.03)
  plt.hist(fret_array, density=True, bins= bin, color = hist_color, alpha = 0.6)
  # plt.plot(x_axis, y_axis0+y_axis1+y_axis2, color = '#000000', lw=3, ls='dashed')
  plt.xlim(0,1) 
  plt.xlabel(r"FRET", fontsize=20)
  plt.ylabel(r"Probability Density", fontsize=20)
  plt.savefig((datapath / f'{filestr}_gaussian mixture'),
               dpi = 1200, 
               bbox_inches = 'tight'
               ) 
  
  plt.show()
  plt.close('all')





def main():

  datapath = Path(r'D:\CWH\2023\20230703\1_DNA FRET_aoi')
  filestr = 'g_combined'
  trace_path= (datapath / (filestr + "_traces.npz"))
  category_path = (datapath / (filestr + "_category.npy"))
  color = ['#4F9D9D', '#7373B9', '#C48888','#D04978']
  hist_color = ['#8CAD05']
  fret_list = []
  traces = ts.TimeTraces.from_npz_eb(trace_path)
  category = np.load(category_path) # 1D ndarray




  for i, analyzable in enumerate(category):
    if not analyzable:
      continue
    donor_intensity = traces.get_intensity(imp.Channel('green', 'green'), i)
    acceptor_intensity = traces.get_intensity(imp.Channel('green', 'red'),i)
    total = donor_intensity + acceptor_intensity
    fret = acceptor_intensity / total
    w = 20
    avg_fret = np.mean(fret[:w])
    if avg_fret > 1:
      continue
    else:
      fret_list.append(avg_fret)

  fret_list = np.array(fret_list)
  fret_array = np.array(fret_list)
  fret_array = fret_array.reshape(-1,1)
  # print(fret_array)

# bic
  bics = []
  min_bic = 0
  counter=1
  for i in range (10): # test the AIC/BIC metric between 1 and 10 components
    gmm = GMM(n_components = counter, 
              max_iter=1000, 
              random_state=0, 
              covariance_type = 'full'
              ).fit(fret_array)
    bic = gmm.bic(fret_array)
    bics.append(bic)
    if bic < min_bic or min_bic == 0:
      min_bic = bic
      opt_bic = counter
    counter = counter + 1


  # plot the evolution of BIC/AIC with the number of components
  fig = plt.figure(figsize=(10, 4))
  ax = fig.add_subplot(1,2,1)
  # Plot 1
  plt.plot(np.arange(1,11), bics, 'o-', lw=3, c='black', label='BIC')
  plt.legend(frameon=False, fontsize=15)
  plt.xlabel('Number of components', fontsize=20)
  plt.ylabel('Information criterion', fontsize=20)
  plt.xticks(np.arange(0,11, 2))
  plt.title('Opt. components = '+str(opt_bic), fontsize=20)
  plt.savefig((datapath / f'{filestr}_bset_component'), 
              dpi = 1200, 
              bbox_inches = 'tight'
              )
  
  plt.show()
  plt.close()

  n = 1
  gmm = GMM(n_components = n, max_iter=1000, random_state=10, covariance_type = 'full')
  
  
  # find useful parameters
  mean = gmm.fit(fret_array).means_  
  covs  = gmm.fit(fret_array).covariances_
  weights = gmm.fit(fret_array).weights_
  print('mean:',mean,'covs:' ,covs, 'weights:', weights)

  # create necessary things to plot
  x_axis = np.arange(0, 1, 0.01)
  plot_his(x_axis, mean, covs, weights, fret_array, 
        color, datapath, filestr, hist_color)






if __name__ == '__main__':
  main()