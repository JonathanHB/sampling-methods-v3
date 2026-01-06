import importlib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

import MSM_methods
import auxilliary_MSM_methods

import utility_v1
import propagators_v2
import energy_landscapes_v1

import parallel_simulations_v2
import weighted_ensemble_v2
import metadynamics_v1
import multiple_walker_metadynamics_v2
import mtd_estimators
import we_mtd_v1

import visualization_v1

import sys


#------------------------------system----------------------------------
system1 = energy_landscapes_v1.variable_sine_well(scale_height=float(sys.argv[1]))

# system1.plot_quantity(system1.potential)
# system1.plot_quantity(system1.F)
# plt.show()

kT = 1     #Boltzmann factor
dt = 0.01  #integration timestep

#TODO make the parameters below and system (and method hyperparams) into a TEST_SETUP object so we can reproduce ones that worked well
#hyperparameters for estimating convergence
n_timepoints = 20
n_bootstrap = 3

#binning parameters
n_analysis_bins = 100
#save frequency

#simulation time limits (integration steps)
#aggregate_simulation_limit = 2420000 #150000000

#number of parallel simulations that are run at a time.
#this is the maximum number in the case of WE simulations.
# If a WE run requires more than this number of this walkers per WE round, 
# they would have to be run serially and the WE round would 
# consume multiple WE round lengths worth of molecular time.
# the WE implementation in this sofware package does not actually do this under the hood, 
# but calculates how much molecular time would have been consumed if the WE run had been executed on n_parallel GPUs
n_parallel = 256

#note: system can diffuse to uniformly fill hexic well in 50k steps
#so sampling in simulations of at least that timescale is definitely limited by energy barriers rather than being diffusion-limited
#could set this time to equal the time it took 50% of walkers to diffuse across a flat landscape of width equal to the test landscape
molecular_time_limit = 100000
#100000 steps is enough to sample the deep sine well except for the highest two metastable states at the ends


#how often the compute nodes or simulation processes can communicate
#really there should be some linear tradeoff (or tradeoff proportional to amount of data which needs to be communicated) between this and molecular_time_limit
min_communication_interval = 100 #simulation steps; current value is probably outlandishly short
#this could be made shorter but does not appear to be a limiting factor
min_frame_save_interval = min_communication_interval

#total amount of frames each method is allowed to save
#this does not appear to be a limiting factor
#frame_save_limit = 100000

#move the number of WE rounds out here; number of data points should be held constant across methods because it reflects lab storage capacity
#we_rounds_per_timepoint = 10 #convert to process communication frequency
#save_period = int(round(molecular_time_limit/(we_rounds_per_timepoint*n_timepoints))) #in steps

#define bins and calculate the true populations thereof
binbounds, bincenters, step = system1.analysis_bins_1d(n_analysis_bins)

true_populations, true_energies = system1.normalized_pops_energies(kT, bincenters)

system_args = system1, kT, dt
resource_args = n_parallel, molecular_time_limit, min_communication_interval, min_frame_save_interval
bin_args = n_timepoints, n_analysis_bins, binbounds, bincenters
true_values = true_populations, true_energies

coverage_thresh = 0.5
RMS_energy_error_thresh = 1


n_samplers = 3

bs_times = [[] for _ in range(n_samplers)]

bs_mean_coverage = [[] for _ in range(n_samplers)]
bs_std_coverage = [[] for _ in range(n_samplers)]

bs_mean_error = [[] for _ in range(n_samplers)]
bs_std_error = [[] for _ in range(n_samplers)]

bs_mean_error_le = [[] for _ in range(n_samplers)]
bs_std_error_le = [[] for _ in range(n_samplers)]

obs_names = [[] for _ in range(n_samplers)]
obs_trjs = [[] for _ in range(n_samplers)]

walkers_per_bin = 10

j=0
sampler_params_we_mtd = [walkers_per_bin, n_analysis_bins, 10, [0.5], 0.001, 1, "we_mtd"]
we_mtd_convergence_times, bs_times[j], bs_mean_coverage[j], bs_std_coverage[j], bs_mean_error[j], bs_std_error[j], bs_mean_error_le[j], bs_std_error_le[j], obs_names[j], obs_trjs[j] = utility_v1.time_to_coverage_accuracy(coverage_thresh, RMS_energy_error_thresh, n_bootstrap, system_args, resource_args, bin_args, we_mtd_v1.sampler_we_mtd, sampler_params_we_mtd, true_values, sys.argv[1])

j=1
sampler_params_we = [walkers_per_bin, n_analysis_bins, 1, "we"]
we_convergence_times, bs_times[j], bs_mean_coverage[j], bs_std_coverage[j], bs_mean_error[j], bs_std_error[j], bs_mean_error_le[j], bs_std_error_le[j], obs_names[j], obs_trjs[j] = utility_v1.time_to_coverage_accuracy(coverage_thresh, RMS_energy_error_thresh, n_bootstrap, system_args, resource_args, bin_args, weighted_ensemble_v2.sampler_we, sampler_params_we, true_values, sys.argv[1])

j=2
sampler_params_parallel_mwm = [10, [0.5], 0.001, 2, "mtd"]
parallel_mwm_convergence_times, bs_times[j], bs_mean_coverage[j], bs_std_coverage[j], bs_mean_error[j], bs_std_error[j], bs_mean_error_le[j], bs_std_error_le[j], obs_names[j], obs_trjs[j] = utility_v1.time_to_coverage_accuracy(coverage_thresh, RMS_energy_error_thresh, n_bootstrap, system_args, resource_args, bin_args, multiple_walker_metadynamics_v2.sampler_parallel_hist_mtd, sampler_params_parallel_mwm, true_values, sys.argv[1])

# j=3
# sampler_params_parallel = [2, "parallel"]
# parallel_convergence_times, bs_times[j], bs_mean_coverage[j], bs_std_coverage[j], bs_mean_error[j], bs_std_error[j] = utility_v1.time_to_coverage_accuracy(coverage_thresh, RMS_energy_error_thresh, n_bootstrap, system_args, resource_args, bin_args, parallel_simulations_v2.sampler_parallel_hist, sampler_params_parallel, true_values)


legend = [o for obsn in obs_names for o in obsn] #["mwm+we msm", "we msm", "mwm grid+hist", "parallel msm"]

plt.clf()
for j in range(n_samplers):
    for i in range(len(bs_times[j])):
        print(i,j)
        plt.errorbar(bs_times[j][i], bs_mean_coverage[j][i], yerr=bs_std_coverage[j][i])

plt.title("coverage vs time")
plt.ylim(0,1)
plt.xlim(0, molecular_time_limit)

plt.xlabel("molecular time")
plt.ylabel("fractional coverage")
plt.legend(legend)

plt.savefig(f"figures/coverage_vs_time_well_depth_scale_{sys.argv[1]}.png", format="png", dpi=600)
plt.clf()#show()


plt.clf()
for j in range(n_samplers):
    for i in range(len(bs_times[j])):
        plt.errorbar(bs_times[j][i], bs_mean_error[j][i], yerr=bs_std_error[j][i])

plt.title("error vs time")
plt.ylim(0,3)
plt.xlim(0, molecular_time_limit)

plt.xlabel("molecular time")
plt.ylabel("RMS energy error (kT)")
plt.legend(legend)

plt.savefig(f"figures/error_vs_time_well_depth_scale_{sys.argv[1]}.png", format="png", dpi=600)
plt.clf()



colors = ["red", "orange", "green", "cyan", "blue", "purple"]
ncolor = 0

for j in range(n_samplers):
    agg_time_trjs, mol_time_trjs, error_trjs, error_trjs_le, coverage_trjs = obs_trjs[j]

    for method_i in range(len(agg_time_trjs)):
        for replicate_i in range(len(agg_time_trjs[0])):
            if replicate_i == 0:
                plt.plot(agg_time_trjs[method_i][replicate_i], coverage_trjs[method_i][replicate_i], color = colors[ncolor])
            else:
                plt.plot(agg_time_trjs[method_i][replicate_i], coverage_trjs[method_i][replicate_i], color = colors[ncolor], label='_nolegend_')
        
        ncolor += 1
    # for i in range(len()):
    #     plt.plot()

plt.title("coverage vs time")
plt.ylim(0,1)
plt.xlim(0, molecular_time_limit*n_parallel)

plt.xlabel("aggregate time (frames)")
plt.ylabel("fractional coverage")
plt.legend(legend)

plt.savefig(f"figures/coverage_vs_agg_time_well_depth_scale_{sys.argv[1]}.png", format="png", dpi=600)
plt.clf()

ncolor = 0

for j in range(n_samplers):
    agg_time_trjs, mol_time_trjs, error_trjs, error_trjs_le, coverage_trjs = obs_trjs[j]

    for method_i in range(len(agg_time_trjs)):
        for replicate_i in range(len(agg_time_trjs[0])):
            if replicate_i == 0:
                plt.plot(agg_time_trjs[method_i][replicate_i], error_trjs[method_i][replicate_i], color = colors[ncolor])
            else:
                plt.plot(agg_time_trjs[method_i][replicate_i], error_trjs[method_i][replicate_i], color = colors[ncolor], label='_nolegend_')
        
        ncolor += 1
    # for i in range(len()):
    #     plt.plot()

plt.title("energy error")
#plt.ylim(0,1)
plt.xlim(0, molecular_time_limit*n_parallel)

plt.xlabel("aggregate time (frames)")
plt.ylabel("RMS energy error (kT)")
plt.legend(legend)

plt.savefig(f"figures/error_vs_agg_time_well_depth_scale_{sys.argv[1]}.png", format="png", dpi=600)
plt.clf()


ncolor = 0

for j in range(n_samplers):
    agg_time_trjs, mol_time_trjs, error_trjs, error_trjs_le, coverage_trjs = obs_trjs[j]

    for method_i in range(len(agg_time_trjs)):
        for replicate_i in range(len(agg_time_trjs[0])):
            if replicate_i == 0:
                plt.plot(agg_time_trjs[method_i][replicate_i], error_trjs_le[method_i][replicate_i], color = colors[ncolor])
            else:
                plt.plot(agg_time_trjs[method_i][replicate_i], error_trjs_le[method_i][replicate_i], color = colors[ncolor], label='_nolegend_')
        
        ncolor += 1
    # for i in range(len()):
    #     plt.plot()

plt.title("low energy state error")
#plt.ylim(0,1)
plt.xlim(0, molecular_time_limit*n_parallel)

plt.xlabel("aggregate time (frames)")
plt.ylabel("RMS energy error (kT)")
plt.legend(legend)

plt.savefig(f"figures/le_error_vs_agg_time_well_depth_scale_{sys.argv[1]}.png", format="png", dpi=600)
plt.clf()