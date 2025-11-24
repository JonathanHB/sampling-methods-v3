from collections import Counter
import numpy as np
import MSM_methods
import time

import propagators_v2
import utility_v1

import matplotlib.pyplot as plt

#run parallel trajectories and estimate the energy landscape by making a histogram of all the frames
def parallel_trj_histogram(state, params):
    
    #unpack inputs
    trjs, cumulative_aggregate_time, cumulative_molecular_time = state #this is unnecessarily complicated but keeps the same structure as other methods
    system, kT, dt, nsegs, save_period, n_parallel, binbounds, bincenters = params
    
    #run dynamics
    new_trjs = propagators_v2.propagate(system, kT, trjs[-1].copy(), dt, nsegs, save_period)
    trjs = np.concatenate((trjs, new_trjs), axis = 0)

    #----------------------------histogram-based population estimation----------------------------#

    #unlike np.digitize, np.histogram does not include data past the ends of the bin range, so we add artificial end bin boundaries
    binbounds_ends = [-999]+[bb for bb in binbounds]+[999]

    #estimate state populations from histogram
    #this will have to be replaced with a binning function that works for higher dimensions. 
    # It may make sense to abstract the binner object from WE into utility and use it here too
    est_bin_pops = np.histogram(trjs.flatten(), binbounds_ends, density=False)
    est_bin_pops_norm = [ebp/len(trjs.flatten()) for ebp in est_bin_pops[0]]

    #----------------------------MSM-based population estimation----------------------------------#
    
    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 
    
    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    # and then reshaping to combine transitions from all parallel trajectories
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:]))
    transitions = transitions.reshape((2, transitions.shape[1]*transitions.shape[2])).transpose()

    #build MSM
    eqp_msm = MSM_methods.transitions_to_eq_probs_v2(transitions, len(binbounds)+1, show_TPM=False)
    
    cumulative_molecular_time += nsegs*save_period
    cumulative_aggregate_time += n_parallel*nsegs*save_period

    return (trjs, cumulative_aggregate_time, cumulative_molecular_time), (cumulative_aggregate_time, cumulative_molecular_time, est_bin_pops_norm, eqp_msm), False


#set up and run parallel simulations and estimate the energy landscape with a histogram
def sampler_parallel_hist(system_args, resource_args, bin_args, sampler_params):

    #----------------------------------input handling--------------------------------

    system, kT, dt = system_args
    n_parallel, molecular_time_limit, min_communication_interval, save_period = resource_args
    n_timepoints, n_analysis_bins, binbounds, bincenters = bin_args

    #determine number of steps for each parallel simulation per timepoint
    nsteps = int(round(molecular_time_limit/n_timepoints))
    #number of frames to save for each parallel simulation per timepoint
    # = number of simulation segments of length save_period to run per timepoint
    nsegs = int(round(nsteps/save_period))

    #molecular and aggregate times accounting for rounding
    actual_molecular_time = nsegs*save_period*n_timepoints
    actual_aggregate_time = n_parallel*actual_molecular_time

    print("\n")
    print("---------------------PARALLEL---------------------")
    print(f"running {n_parallel} parallel simulations")
    print(f"molecular time: {actual_molecular_time} steps;  aggregate time: {actual_aggregate_time} steps")
    print(f"data points saved: {nsegs*n_timepoints*n_parallel} at {save_period}-step intervals")

    #--------------------------------set up and run system-----------------------------

    #initiate all simulations in the same state
    trjs = np.array([system.standard_init_coord for element in range(n_parallel)]).reshape((1, n_parallel, len(system.standard_init_coord)))
    #long_trj_inds = np.array([system.standard_init_index for element in range(n_parallel)]).reshape((n_parallel, 1))

    t1 = time.time()
    #pack the initial state and parameters and run dynamics
    initial_state = (trjs, 0, 0)
    params = (system, kT, dt, nsegs, save_period, n_parallel, binbounds, bincenters)
    observables_over_time_transposed = utility_v1.run_for_n_timepoints(parallel_trj_histogram, params, initial_state, n_timepoints)
    
    t2 = time.time()
    #print(f"simulation={t2-t1}")

    #effectively transpose the list of lists so the first axis is observable type rather than time
    #but without the data type/structure requirement of a numpy array
    observables_over_time = [list(row) for row in zip(*observables_over_time_transposed)]

    return observables_over_time

