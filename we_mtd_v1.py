import numpy as np

import MSM_methods
import metadynamics_v1
import weighted_ensemble_v2
import propagators_v2
import utility_v1

import mtd_estimators


#run dynamics and return the results
#for arguments and returns see the comments in propagators.py
#this version of this method exists to store the variables defined in __init__() without cluttering up the weighted_ensemble() function, not to do anything new
#other versions will have metadynamics grids and updating functions 
#   TODO define a fancier version that uses a metadynamics grid
class we_mtd_propagator_1():
    
    def __init__(self, system, kT, timestep, nsteps, grid):
        self.system = system
        self.kT = kT
        self.timestep = timestep
        self.nsteps = nsteps
        self.grid = grid
        #self.save_period = save_period
    
    def propagate(self, x, w):
        return (propagators_v2.propagate_mtd_save1(self.system, self.kT, x, self.timestep, self.nsteps, self.grid, w))
    
    def mtd_grid(self):
        return None


#a wrapper for running weighted ensemble in segments and feeding them through msm_trj_analysis.run_for_n_timepoints()
def we_mtd_histogram(state, params):
    
    #TODO: add support for different WE and analysis bins, which is currently (and deceptively) half done
    
    #unpack inputs
    x, e, w, cb, b, propagator, cumulative_observables, cumulative_aggregate_time, cumulative_molecular_time = state
    split_merge, config_binner, bincenters, ensemble_classifier, binner, calc_observables, nrounds, walkers_per_bin, aggregate_simulation_limit = params

    #run dynamics
    x, e, w, cb, b, propagator, new_observables = weighted_ensemble_v2.weighted_ensemble(x, e, w, cb, b, propagator, split_merge, config_binner, ensemble_classifier, binner, calc_observables, nrounds, walkers_per_bin)

    #update cumulative observables and aggregate time
    observables = cumulative_observables+new_observables
    cumulative_molecular_time += nrounds*propagator.nsteps
    cumulative_aggregate_time += sum([nobs[-1] for nobs in new_observables])*propagator.nsteps
    #cumulative_agg_t += sum([nobs[-1] for nobs in new_observables])*propagator.nsteps

    #----------------------------histogram-based population estimation----------------------------#

    #estimate state populations
    cumulative_config_bins = np.concatenate([o[0] for o in observables[1:]], axis = 1).transpose()

    pops_hist = np.zeros(config_binner.n_bins)  #initialize estimated bin populations to 0
    for cbi in cumulative_config_bins:
        pops_hist[int(cbi[0])] += cbi[1]

    pops_hist /= np.sum(pops_hist)  #normalize estimated bin populations


    #----------------------------MSM-based population estimation----------------------------#

    aggregate_transitions = np.concatenate([o[2] for o in observables[1:]], axis = 1).transpose()
    eqp_msm = MSM_methods.transitions_to_eq_probs_v2(aggregate_transitions, config_binner.n_bins, weights=None, show_TPM=False)


    #----------------------------mtd-weighted MSM-based population estimation----------------------------#

    mtd_transition_weights = np.concatenate([o[5] for o in observables[1:]])
    eqp_msm_weighted = MSM_methods.transitions_to_eq_probs_v2(aggregate_transitions, config_binner.n_bins, weights=mtd_transition_weights, show_TPM=False)


    #------------------------estimate populations from metadynamics grid-------------------------#

    pops_grid = np.exp(propagator.grid.grid) #((kT+propagator.grid.dT)/propagator.grid.dT)*
    pops_grid /= np.sum(pops_grid)


    #-----------------------------estimate populations from grid in sampled bins only

    pops_grid_masked = [pg if h>0 else 0 for pg, h in zip(pops_grid, pops_hist)]
    pops_grid_masked /= np.sum(pops_grid_masked)


    #----------------------------combined grid+histogram-based population estimation----------------------------#

    cumulative_mtd_weights = np.concatenate([o[4] for o in observables[1:]])

    pops_hist_mtd = np.zeros(config_binner.n_bins)  #initialize estimated bin populations to 0
    for cbi, cmtdw in zip(cumulative_config_bins, cumulative_mtd_weights):
        pops_hist_mtd[int(cbi[0])] += cmtdw

    pops_hist_mtd /= np.sum(pops_hist_mtd)  #normalize estimated bin populations
    

    #----------------------------combined grid+histogram+we weight-based population estimation----------------------------#
    #this did not work well because WE weights are too noisy
    
    cumulative_mtd_weights = np.concatenate([o[5] for o in observables[1:]])

    pops_hist_we_mtd = np.zeros(config_binner.n_bins)  #initialize estimated bin populations to 0
    for cbi, cmtdw in zip(cumulative_config_bins, cumulative_mtd_weights):
        pops_hist_we_mtd[int(cbi[0])] += cmtdw*cbi[1]

    pops_hist_we_mtd /= np.sum(pops_hist_we_mtd)  #normalize estimated bin populations

    #----------------------------franken MSM----------------------------#

    stacked_transitions = [o[2].transpose() for o in observables[1:]] #the number of transitions is not constant so these cannot be stacked
    stacked_grid_weights = np.stack([o[6] for o in observables])

    print(stacked_grid_weights.shape)

    msm_v3_pops_all = mtd_estimators.MSM_v3(stacked_transitions, config_binner.binbounds, stacked_grid_weights, propagator.system, bincenters, propagator.kT)

    return (x, e, w, cb, b, propagator, observables, cumulative_aggregate_time, cumulative_molecular_time), (cumulative_aggregate_time, cumulative_molecular_time, pops_hist_mtd, eqp_msm_weighted, msm_v3_pops_all), cumulative_aggregate_time >= aggregate_simulation_limit


########################################  MAIN SAMPLER CLASS  ########################################

def sampler_we_mtd(system_args, resource_args, bin_args, sampler_params):

    #----------------------------------input handling--------------------------------

    system, kT, dt = system_args
    n_parallel, molecular_time_limit, min_communication_interval, save_period = resource_args
    n_timepoints, n_analysis_bins, binbounds, bincenters = bin_args #TODO these should be used
    walkers_per_bin, n_we_bins, dT, stdev, rate_per_frame = sampler_params

    binbounds_we, bincenters_we, step_we = system.analysis_bins_1d(n_we_bins)

    #determine number of steps for each parallel simulation per timepoint
    nsteps = int(round(molecular_time_limit/n_timepoints))
    #number of frames to save for each parallel simulation per timepoint
    # = number of simulation segments of length save_period to run per timepoint
    n_rounds_per_timepoint = int(round(nsteps/min_communication_interval))

    #molecular and aggregate times accounting for rounding
    actual_molecular_time = n_rounds_per_timepoint*min_communication_interval*n_timepoints
    max_actual_aggregate_time = n_parallel*actual_molecular_time

    # Adjust gaussian height so that potential energy is added at a constant rate 
    # regardless of the gaussian deposition frequency.
    # Because gaussians are weighted by the walker weights, which are normalized, 
    # we do not divide by the number of parallel simulations (which varies anyway in WE).
    rate = rate_per_frame*save_period

    print("\n")
    print("---------------------WEIGHTED ENSEMBLE + MULTIPLE WALKER METADYNAMICS---------------------")
    print(f"weighted ensemble with {walkers_per_bin} walkers per bin in {len(binbounds_we)+1} bins for {n_rounds_per_timepoint*n_timepoints} WE rounds of {min_communication_interval} steps each")
    print(f"molecular time: {actual_molecular_time} steps;  maximum aggregate time: {max_actual_aggregate_time} steps")
    print(f"maximum data points saved: {n_rounds_per_timepoint*n_timepoints*walkers_per_bin*(len(binbounds_we)+1)} at {min_communication_interval}-step intervals")
    print(f"gaussians of height {rate} are added every {save_period} steps")

    #--------------------------------set up and run system-----------------------------

    #initialize instances of classess
    config_binner = weighted_ensemble_v2.config_binner_1(binbounds_we)
    ensemble_classifier = weighted_ensemble_v2.ensemble_classifier_1(system.macro_class_parallel)
    binner = weighted_ensemble_v2.binner_1()

    #grid and metadynamics propagator
    grid = metadynamics_v1.grid(system.standard_analysis_range, len(binbounds)-1, rate, dT, stdev)
    propagator0 = we_mtd_propagator_1(system, kT, dt, min_communication_interval, grid) #the propagator evolves over time in the case of metadynamics

    #initial state
    x0 = np.array([system.standard_init_coord for element in range(walkers_per_bin)]) #.reshape((walkers_per_bin, 1, len(system.standard_init_coord)))
    e0 = [system.macro_class(x0i) for x0i in x0] #initial ensemble is determined by the macrostate classifier
    w0 = [1/walkers_per_bin for element in range(walkers_per_bin)]
    cb0 = config_binner.bin(x0)  #configurational bins
    b0 = binner.bin(cb0, e0)
    #prop_out_0 = [1 for element in range(walkers_per_bin)]

    init_grid_weights = grid.grid_weights(kT)

    cumulative_observables0 = [([],[],[],[],[],[], init_grid_weights, 0)]  #list of lists; each sublist contains the observables calculated at each WE round

    #pack the initial state and parameters and run dynamics
    initial_state = (x0, e0, w0, cb0, b0, propagator0, cumulative_observables0, 0, 0) #the final 0 is the initial aggregate simulation time
    params = (weighted_ensemble_v2.split_merge, config_binner, bincenters, ensemble_classifier, binner, weighted_ensemble_v2.calc_observables_1, n_rounds_per_timepoint, walkers_per_bin, max_actual_aggregate_time)
    time_x_observables = utility_v1.run_for_n_timepoints(we_mtd_histogram, params, initial_state, n_timepoints)

    #effectively transpose the list of lists so the first axis is observable type rather than time
    #but without the data type/structure requirement of a numpy array
    observables_x_time = [list(row) for row in zip(*time_x_observables)]

    final_aggregate_time = observables_x_time[0][-1]
    print(f"aggregate simulation time: {final_aggregate_time} steps")
    print(f"aggregate number of walkers = number of data points saved = {final_aggregate_time/min_communication_interval} at {min_communication_interval}-step intervals")

    estimate_names = ["weighted histogram", "weighted msm", "franken MSM"] #["histogram", "msm", "weighted msm", "grid_masked", "grid + histogram"]#, "grid + histogram + we"]

    return observables_x_time, estimate_names