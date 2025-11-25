#run a sampling/analysis method and save out its progress at n_timepoints equally spaced* points
# *equally spaced by wall clock time
# parameters:
#   analysis_method: the sampling and analysis method to analyze
#   params: parameters for the analysis method, including the wall clock time per time point
#   initial_state: initial state of the system, including both coordinates and other inputs such as the metadynamics potential grid
#   n_timepoints: number of serial simulation segments to run
# returns:
#   a list of the observables calculated by the method at each time point

def run_for_n_timepoints(analysis_method, params, initial_state, n_timepoints):

    method_state = initial_state
    method_output = []

    for t in range(n_timepoints):
        print(f"running segment {t+1} of {n_timepoints}", end='\r')
        method_state, observable, stopflag = analysis_method(method_state, params)
        method_output.append(observable)
        if stopflag:
            print("method halted by exceeding aggregate simulation limit")
            break

    return method_output


#-------------------------------------------------------------------------------------------------------------
#bootstrapping method
#-------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import visualization_v1

def time_to_coverage_accuracy(coverage_thresh, RMS_energy_error_thresh, n_bootstrap, system_args, resource_args, bin_args, sampler, sampler_params, true_values):
    
    system, kT, dt = system_args
    n_parallel, molecular_time_limit, min_communication_interval, min_frame_save_interval = resource_args
    n_timepoints, n_analysis_bins, binbounds, bincenters = bin_args
    true_populations, true_energies = true_values

    #bootstrap to estimate convergence times
    #TODO this is not general; each sampler should have a n_methods variable that the bootstrapper can use to make a 2d array
    #this will probably entail making the sampler a class (and then making the associated method run by run_for_n_timepoints into a method of that class)
    n_methods=sampler_params[-1]
    convergence_times = np.zeros((n_methods,n_bootstrap))
    #convergence_times_MSM = np.zeros(n_bootstrap)

    for bi in range(n_bootstrap):
        print(f"bootstrap round {bi+1}\n")
        observables_over_time, observable_names = sampler(system_args, resource_args, bin_args, sampler_params[:-1])

        molecular_times = observables_over_time[1]
        aggregate_times = observables_over_time[0]

        #look at convergence of each observable
        for oi, observable in enumerate(observables_over_time[2:]):

            energies = -kT*np.log(observable)
            visualization_v1.plot_landscape_estimate(bincenters, energies, true_energies, observable_names[oi], xrange = (-22,20), yrange = (-5,25))

            RMS_energy_errors = []
            coverages = [] # this is the fraction of PC space within bin_width/2 of a sampled configuration

            #calculate landscape coverage and energy error at each timepoint
            for ti, populations in enumerate(observable):
            
                energies = -kT*np.log(populations)
                RMS_energy_error = np.sqrt(np.mean([(e-te)**2 for e, te, p in zip(energies, true_energies, populations) if p > 0]))
                RMS_energy_errors.append(RMS_energy_error)

                coverage = sum([1 if p > 0 else 0 for p in populations])/len(populations)
                coverages.append(coverage)
                
                if convergence_times[oi,bi] == 0 and coverage >= coverage_thresh and RMS_energy_error <= RMS_energy_error_thresh:
                    convergence_times[oi,bi] = molecular_times[ti]

            plt.plot(molecular_times, RMS_energy_errors)
            plt.plot(molecular_times, coverages)
            plt.legend(["RMS energy error (kT)", "fractional landscape coverage"])
            plt.xlabel("molecular time")
            plt.ylabel("energy error (kT) or\nlandscape coverage (dimensionless)")
            plt.title(observable_names[oi])
            plt.ylim(0,2)
            plt.xlim(0, molecular_time_limit)
            plt.show()

            plot_agg_vs_mol_t = False
            if plot_agg_vs_mol_t:
                plt.plot(molecular_times, aggregate_times)
                plt.xlabel("molecular time")
                plt.ylabel("aggregate time")
                plt.title(observable_names[oi])
                plt.show()

    print(convergence_times)

    for c in convergence_times:
        print(np.mean(c))
        print(np.std(c))