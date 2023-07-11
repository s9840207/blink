import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from pathlib import Path
from scipy.stats import norm
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mpl
import blink.time_series as ts
import blink.image_processing as imp

def fret_data(datapath, filestr, channel):

# datapath = Path(r'D:\CWH\20230602\3_DNA+RAD51+BCDX2 real time FRET_aoi')
# filestr = 'g_combined'
    trace_path= (datapath / (filestr + "_traces.npz"))
    category_path = (datapath / (filestr + "_category.npy"))
    traces = ts.TimeTraces.from_npz(trace_path)
    category = np.load(category_path) # 1D ndarray
    frame = traces.get_time(channel)
    fret_list = []
    for molecule, analyzable in enumerate(category):
        if not analyzable:
            continue
        donor_intensity = traces.get_intensity(imp.Channel('green', 'green'), molecule)
        acceptor_intensity = traces.get_intensity(imp.Channel('green', 'red'),molecule)
        total = donor_intensity + acceptor_intensity
        fret = acceptor_intensity / total
        condtition = fret > 1
        fret = fret[condtition]
        fret_list.append(fret)
        fret_array = np.array(fret_list)
    return fret_array
def bic(datapath, filestr, traces):
    bics = []
    min_bic = 0
    counter=1
    for n in range(len(traces)):
        fitted_traces = traces[n]
        fitted_traces = traces.reshape(-1,1)
        for i in range (10): # test the AIC/BIC metric between 1 and 10 components
            model = hmm.GaussianHMM(n_components = counter, n_iter = 50, random_state = 0)
            # labels = hmm.GMMHMM.fit(fret_array).predict(fret_array)
            
            model.fit(fitted_traces)
            bic = model.bic(fitted_traces)
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
        plt.savefig((datapath / f'{n}_bset_component'), dpi = 1200, bbox_inches = 'tight')
        
        plt.show()
        plt.close()
    



def main():
    # Step 1: Prepare your data and train the HMM model
    datapath = Path(r'D:\CWH\20230609\3_DNA+BCDX2+RAD51 0-5min real time_aoi')
    filestr = 'g_combined'
    channel = imp.Channel('green', 'red')
    trace_path= (datapath / (filestr + "_traces.npz"))
    category_path = (datapath / (filestr + "_category.npy"))
    raw_traces = ts.TimeTraces.from_npz_eb(trace_path)
    category = np.load(category_path) # 1D ndarray
    time= raw_traces.get_time(channel)
    fret_list = []
    for molecule, analyzable in enumerate(category):
        if analyzable:
        
            donor_intensity = raw_traces.get_intensity(
                imp.Channel('green', 'green'), 
                molecule
                )
            acceptor_intensity = raw_traces.get_intensity(
                imp.Channel('green', 'red'),
                molecule
                )
            total = donor_intensity + acceptor_intensity
            fret = acceptor_intensity / total
            fret_list.append(fret)
            traces = np.array(fret_list)
        
    
    
    for n in range(len(traces)):
        bics = []
        min_bic = 0
        counter=1
        fitted_traces = traces[n].reshape(-1,1)
        
        for i in range (10): # test the AIC/BIC metric between 1 and 10 components
            model = hmm.GaussianHMM(n_components = counter,
                                     n_iter = 1000,
                                     random_state = 10)
            # labels = hmm.GMMHMM.fit(fret_array).predict(fret_array)
            
            model.fit(fitted_traces)
            bic = model.bic(fitted_traces)
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
        plt.savefig((datapath / f'{n}_bset_component'), 
                    format = 'png',
                    dpi = 1200, 
                    bbox_inches = 'tight')
        
        plt.show()
        plt.close()

        model = hmm.GaussianHMM(n_components = 2, n_iter = 50, random_state = 0) 
        model.fit(fitted_traces)
        hidden_states = model.predict(fitted_traces)
        state_values = model.means_
        
        transitions = np.where(np.diff(hidden_states) != 0)[0] + 1
        transition_line = np.repeat(state_values[hidden_states], 1)
        
        
        
        plt.figure(figsize=(15,6))
        plt.plot(time, fitted_traces, color = 'blue', label = 'trace')
        plt.step(np.arange(len(transition_line))*0.1,
                transition_line,
                color='red',
                label='HHM')
        plt.xlabel('time(s)')
        plt.ylabel('FRET')
        plt.ylim(0,1)
        plt.xlim(0,360)

        plt.legend()
        
        
        plt.savefig(datapath / f"{n}_hmm_fitting.png", 
                    format="png", 
                    dpi=1200,
                    bbox_inches = 'tight')
        plt.show()
        plt.close()
        
if __name__ == '__main__':
    main()

