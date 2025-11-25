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
    n_methods = sampler_params[-2]
    sampling_method_name = sampler_params[-1]

    convergence_times = np.zeros((n_methods,n_bootstrap))

    mol_time_trjs = [[[] for _ in range(n_bootstrap)] for __ in range(n_methods)]
    coverage_trjs = [[[] for _ in range(n_bootstrap)] for __ in range(n_methods)]
    error_trjs = [[[] for _ in range(n_bootstrap)] for __ in range(n_methods)]

    #convergence_times_MSM = np.zeros(n_bootstrap)

    for bi in range(n_bootstrap):
        print(f"bootstrap round {bi+1}\n")
        observables_over_time, observable_names = sampler(system_args, resource_args, bin_args, sampler_params[:-2])

        molecular_times = observables_over_time[1]
        aggregate_times = observables_over_time[0]

        #look at convergence of each observable
        for oi, observable in enumerate(observables_over_time[2:]):

            energies = -kT*np.log(observable)
            savefilename = f"figures/{sampling_method_name}-{observable_names[oi]}-landscape-rep-{bi}.png"

            visualization_v1.plot_landscape_estimate(bincenters, energies, true_energies, observable_names[oi], xrange = (-22,20), yrange = (-5,25), savefilename=savefilename)

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
            
            mol_time_trjs[oi][bi] = molecular_times
            coverage_trjs[oi][bi] = coverages
            error_trjs[oi][bi] = RMS_energy_errors


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


    bs_times = [[] for _ in range(n_methods)]

    bs_mean_coverage = [[] for _ in range(n_methods)]
    bs_std_coverage = [[] for _ in range(n_methods)]

    bs_mean_error = [[] for _ in range(n_methods)]
    bs_std_error = [[] for _ in range(n_methods)]

    for oi in range(n_methods):

        bs_times_ = []
        bs_mean_coverage_ = []
        bs_std_coverage_ = []
        bs_mean_error_ = []
        bs_std_error_ = []

        time_trj_lengths = [len(mol_time_trjs[oi][bi]) for bi in range(n_bootstrap)]
        max_t_round = time_trj_lengths.index(max(time_trj_lengths))

        for t in range(999):
            coverages = [coverage_trjs[oi][bi][t] for bi in range(n_bootstrap) if len(coverage_trjs[oi][bi])>t]
            errors = [error_trjs[oi][bi][t] for bi in range(n_bootstrap) if len(error_trjs[oi][bi])>t]

            if len(coverages) > 1:
                bs_times_.append(mol_time_trjs[oi][max_t_round][t])

                bs_mean_coverage_.append(np.mean(coverages))
                bs_std_coverage_.append(np.std(coverages))
                
                bs_mean_error_.append(np.mean(errors))
                bs_std_error_.append(np.std(errors))
            else:
                break

        # print(bs_times_, bs_mean_coverage_, bs_std_coverage_)

        # plt.errorbar(bs_times_, bs_mean_coverage_, yerr=bs_std_coverage_)
        # plt.title("coverage vs time")
        # plt.ylim(0,1)
        # plt.xlim(0, molecular_time_limit)
        # plt.show()

        # print(bs_times)
        # print(len(bs_times))
        # print(oi)

        bs_times[oi] = bs_times_

        bs_mean_coverage[oi] = bs_mean_coverage_
        bs_std_coverage[oi] = bs_std_coverage_
        bs_mean_error[oi] = bs_mean_error_
        bs_std_error[oi] = bs_std_error_

        # for bi in range(n_bootstrap):
        #     plt.plot(mol_time_trjs[oi][bi], coverage_trjs[oi][bi])
        # plt.ylim(0,1)
        # plt.xlim(0, molecular_time_limit)
        # plt.show()
        
        # # bs_mean_error = []
        # # bs_std_error = []

        # for bi in range(n_bootstrap):
        #     plt.plot(mol_time_trjs[oi][bi], error_trjs[oi][bi])
        # plt.ylim(0,2)
        # plt.xlim(0, molecular_time_limit)
        # plt.show()

    # print(convergence_times)

    # for c in convergence_times:
    #     print(np.mean(c))
    #     print(np.std(c))

    return convergence_times, bs_times, bs_mean_coverage, bs_std_coverage, bs_mean_error, bs_std_error