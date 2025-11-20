from collections import Counter
import numpy as np
import MSM_methods
import time

import propagators_v1
import utility_v1

#run parallel trajectories and estimate the energy landscape by making a histogram of all the frames
def parallel_trj_histogram(state, params):
    
    #unpack inputs
    trjs = state #this is unnecessarily complicated but keeps the same structure as other methods
    system, kT, dt, nsteps, save_period, binbounds, bincenters = params
    
    t3 = time.time()
    #run dynamics
    new_trjs = propagators_v1.propagate(system, kT, trjs[-1].copy(), dt, nsteps, save_period)
    trjs = np.concatenate((trjs, new_trjs), axis = 0)
    t4 = time.time()
    #print(f"dynamics={t4-t3}")
    

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
    
    t2 = time.time()
    #print(f"analysis={t2-t1}")

    return (trjs), (est_bin_pops_norm, eqp_msm), False


#set up and run parallel simulations and estimate the energy landscape with a histogram
def sampler_parallel_hist(system, aggregate_simulation_limit, molecular_time_limit, save_period, n_timepoints, kT, dt, binbounds, bincenters):
    t1 = time.time()
    #determine number of parallel simulations and steps per simulation
    n_parallel = int(round(aggregate_simulation_limit/molecular_time_limit))
    nsteps = int(round(aggregate_simulation_limit/(n_parallel*n_timepoints)))

    print("\n")
    print(f"running {n_parallel} parallel simulations for {nsteps*n_timepoints} steps each")
    print(f"molecular time: {nsteps*n_timepoints} steps;  aggregate time: {nsteps*n_timepoints*n_parallel} steps")
    print(f"data points saved: {aggregate_simulation_limit/save_period} at {save_period}-step intervals")

    #initiate all simulations in the same state
    trjs = np.array([system.standard_init_coord for element in range(n_parallel)]).reshape((1, n_parallel, len(system.standard_init_coord)))
    #long_trj_inds = np.array([system.standard_init_index for element in range(n_parallel)]).reshape((n_parallel, 1))

    t3 = time.time()
    #pack the initial state and parameters and run dynamics
    initial_state = (trjs)
    params = (system, kT, dt, nsteps, save_period, binbounds, bincenters)
    time_x_observables = utility_v1.run_for_n_timepoints(parallel_trj_histogram, params, initial_state, n_timepoints)
    
    t4 = time.time()
    #print(f"simulation={t4-t3}")

    #effectively transpose the list of lists so the first axis is observable type rather than time
    #but without the data type/structure requirement of a numpy array
    observables_x_time = [list(row) for row in zip(*time_x_observables)]
    
    t2 = time.time()
    #print(f"total={t2-t1}")

    return observables_x_time





























#VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV UNREFACTORED CODE BELOW VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

# #TODO does this actually need to be its own method? It has exactly 1 original line of code
# def resume_parallel_simulations_msm(propagator, system, kT, dt, nsteps, save_period, prev_trj, prev_inds):

#     #propagate
#     long_trjs, long_trj_inds = propagator(system, kT, prev_inds[:,-1], dt, nsteps, save_period)

#     #extend the existing trajectory
#     return np.concatenate((prev_trj, long_trjs), axis=1), np.concatenate((prev_inds, long_trj_inds), axis=1) 


# #run parallel trajectories and estimate the energy landscape by making a histogram of all the frames
# def parallel_trj_histogram(state, params):
#     #unpack inputs
#     long_trjs, long_trj_inds = state
#     system, kT, dt, nsteps, save_period, binbounds = params

#     #run dynamics
#     long_trjs, long_trj_inds = resume_parallel_simulations_msm(propagators.propagate_msm, system, kT, dt, nsteps, save_period, long_trjs, long_trj_inds)

#     #actual state populations
#     #bin all the msm states and then add up the equilibrium probabilities of all the states in each bin
#     state_bins = msm_trj_analysis.bin_to_voxels_msmstates(binbounds, system.x)
#     bin_pops = msm_trj_analysis.state_to_bin_populations(state_bins, system.p)

#     #estimated state populations
#     #for this analysis doing a histogram directly would suffice but that will not work for building MSMs
#     #once the trajectory has been binned there is no need to use a histogram
#     binned_trj = msm_trj_analysis.bin_to_voxels_msmtrj(binbounds, system.x, long_trj_inds)
#     bin_counts = Counter(binned_trj.flatten()) #count number of frames in each bin; could be added to bin_to_voxels but would slow down things that don't use it
#     total_counts = len(binned_trj.flatten())
#     est_bin_pops = [bin_counts[sb]/total_counts for sb in np.unique(state_bins)] #estimate populations only for bins containing msm states

#     #calculate the weighted mean absolute error of the estimated bin populations
#     maew = np.mean([spi*abs(espi-spi) for spi, espi in zip(bin_pops, est_bin_pops)])

#     return (long_trjs, long_trj_inds), (maew, est_bin_pops)


# #set up and run parallel simulations and estimate the energy landscape with a histogram
# def sampler_parallel_hist(system, n_parallel, nsteps, save_period, n_timepoints, kT, dt, binbounds):

#     #initiate all simulations in the same state
#     long_trjs = np.array([system.standard_init_coord for element in range(n_parallel)]).reshape((n_parallel, 1, len(system.standard_init_coord)))
#     long_trj_inds = np.array([system.standard_init_index for element in range(n_parallel)]).reshape((n_parallel, 1))

#     #pack the initial state and parameters and run dynamics
#     initial_state = (long_trjs, long_trj_inds)
#     params = (system, kT, dt, nsteps, save_period, binbounds)
#     parallel_trj_outputs = msm_trj_analysis.run_for_n_timepoints(parallel_trj_histogram, params, initial_state, n_timepoints)

#     #unpack outputs
#     est_state_pop_convergence = [i[1] for i in parallel_trj_outputs]
#     maew_convergence = [i[0] for i in parallel_trj_outputs]

#     return est_state_pop_convergence, maew_convergence