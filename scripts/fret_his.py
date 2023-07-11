from pathlib import Path
import numpy as np
import blink.time_series as ts
import blink.image_processing as imp
import matplotlib.pyplot as plt
import scipy.stats
# import rpy2.robjects as robjects
import blink.binding_kinetics as bk
from sklearn.mixture import GaussianMixture


def main():
    path_list = [r'D:\CWH\20230309\3_RPA-DNA FRET_aoi\g1_combined_traces.npz', r'D:\CWH\20230208\dT21+RPA_wash_5min_aoi\g_combined_traces.npz']
    category_path_list = [r'D:\CWH\20230309\3_RPA-DNA FRET_aoi\g1_combined_category.npy', r'D:\CWH\20230208\dT21+RPA_wash_5min_aoi\g_combined_category.npy']
    avg = []
    var = []
    color = ['#96D5DF', '#D94600']
    fitting_line = ['#96D5DF', '#D94600']
    fret_list = [[], []]
    for j, path in enumerate(path_list):
        print(path, j)
        
        path = Path(path)
        traces = ts.TimeTraces.from_npz_eb(path)
        category_path = Path(category_path_list[j]) 
        category = np.load(category_path) # 1D ndarray




        for i, analyzable in enumerate(category):
            if not analyzable:
                continue
            donor_intensity = traces.get_intensity(imp.Channel('green', 'green'), i)
            acceptor_intensity = traces.get_intensity(imp.Channel('green', 'red'),i)
            total = donor_intensity + acceptor_intensity
            fret = acceptor_intensity / total
            w = 10
            avg_fret = np.mean(fret[:w])
            fret_list[j].append(avg_fret)
            print(fret_list)
            # if 0.2< avg_fret <0.5: 
            #     fret_list[j].append(avg_fret)
                

        # fret_list = np.array(fret_list)
        # print(fret_list)
        avg.append(np.mean(fret_list[j]))
        var.append(np.var(fret_list[j]))
        print(avg, var)
        print('-------------------------------------------------------')
        # print(fret_list)
        # From that, we know the shape of the fitted Gaussian.
    # pdf_x0 = np.linspace(np.min(fret_list[0]),np.max(fret_list[0]),100)
    # pdf_y0 = 1.0/np.sqrt(2*np.pi*var[0])*np.exp(-0.5*(pdf_x0-avg[0])**2/var[0])

    # pdf_x1 = np.linspace(np.min(fret_list[1]),np.max(fret_list[1]),100)
    # pdf_y1 = 1.0/np.sqrt(2*np.pi*var[1])*np.exp(-0.5*(pdf_x0-avg[1])**2/var[1])

    plt.hist(fret_list[0], bins = 50, color = color[0], density = True, alpha = 0.6)
    # plt.plot(pdf_x0, pdf_y0, fitting_line[0] )
    plt.xlabel('FRET')
    plt.ylabel('Problibity density')
    plt.xlim(0, 1)
    plt.savefig(Path(r'D:\CWH\20230309\3_RPA-DNA FRET_aoi\RPA-DNA wash FRET', dpi = 1200, bbox_inches = 'tight'))

    # plt.hist(fret_list[1], bins = 30, color = color[0], density = True, alpha = 0.6)
    # plt.plot(pdf_x0, pdf_y0, fitting_line[0])
    # plt.savefig(Path(r'D:\CWH\binding_curve_and_time_trace\FRET\RPA coated dT21 no wash ', dpi = 1200, bbox_inches = 'tight'))

    
        # plt.savefig(r'D:\CWH\binding_curve_and_time_trace\FRET\dT21 FRET (0,72)', dpi = 1200, bbox_inches="tight")
    plt.show()

    # # Then we plot :
    
        



    







if __name__ == '__main__':
    main()