from pathlib import Path
import numpy as np
import blink.binding_kinetics as bk
import blink.time_series as ts
import blink.photobleaching_analysis as pa
from scipy.stats import rankdata



def main():
    path = Path(r'D:\CWH\20221222\1\dT13_binding_gr_inverse_traces.npz')
    traces = ts.TimeTraces.from_npz_eb(path)
    category_path = Path(r'D:\CWH\20221222\1\dT13_binding_gr_inverse_category.npy') 
    category = np.load(category_path) # 1D ndarray 
    channel = ('blue', 'blue')
    state_info = bk.collect_all_channel_state_info(traces)[channel]
    time = traces.get_time(channel)
    #get traces, category, state, and time
    
    number_of_good_traces = 0
    categorizing_traces = {}
    

    #filt analyzable traces
    for i, analyzable in enumerate(category):
        if not analyzable:
            continue
        else:
            sequence = traces.get_viterbi_path(channel, i) 
            intensity = traces.get_intensity(channel, i)

            #simplify trace into state transition
            interval_sequence_transition = [] 
            interval_slices = bk.get_interval_slices(sequence)
            for interval in interval_slices: 
                start = interval.start
                stop = interval.stop
                interval_sequence_transition.append(sequence[start]) 

            # lowest signal belongs to state 0
            state_sequence_transition = rankdata(interval_sequence_transition, method ='dense')-1
            
            first_state_is_nonzero = abs(sequence[0]) > 2 * np.std(intensity[sequence == sequence[0]])
            lowest_state_is_zero = abs(sequence.min()) < 2 * np.std(intensity[sequence == sequence.min()])
            unique = np.unique(state_sequence_transition)
            num_states = len(unique)

            #filt good trace that drops step-wise
            if (first_state_is_nonzero == True) and (lowest_state_is_zero == True):
                drop_to_zero_ar = np.arange(num_states-1, -1, -1) 
                filter_list = state_sequence_transition[0:len(drop_to_zero_ar)] == drop_to_zero_ar
                if all(filter_list) == True:     
                    if num_states == 3:
                        print(f'molecule{i}')                               
                    number_of_good_traces += 1
                    if num_states not in categorizing_traces:
                        categorizing_traces[num_states] = 1
                    else:
                        categorizing_traces[num_states] += 1
                        
            #filt 1 state bound trace
            elif (first_state_is_nonzero == True) and (num_states == 1):
                if num_states not in categorizing_traces:
                    categorizing_traces[num_states] = 1
                else:
                    categorizing_traces[num_states] += 1 

    # calculate the other analyzable traces
    total_num_in_category = 0
    for key, value in categorizing_traces.items():
        total_num_in_category += categorizing_traces[key]
    categorizing_traces['else_num'] = np.count_nonzero(category) - total_num_in_category
    

    # calculate each fraction
    fraction_calculation = {}
    for key, value in categorizing_traces.items():
        fraction_calculation[key] = value / np.count_nonzero(category)


    print(f'number of good traces is {number_of_good_traces}')     
    print(f'categorizing good traces is \n how many states:how many molecules {categorizing_traces}')  
    print(f'fraction_calculation is \n how many states:fraction of good traces {fraction_calculation}')   







if __name__ == '__main__':
    main()