from collections import Counter
import numpy as np

from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components

import metadynamics_v1
import propagators_v2
import utility_v1
import MSM_methods

import matplotlib.pyplot as plt
import time

import sys

import auxilliary_MSM_methods
import mtd_estimators



#run parallel trajectories and estimate the energy landscape by making a histogram of all the frames
def parallel_trj_histogram_mtd(state, params):
    
    #t1 = time.time()

    #unpack inputs
    trjs, weights, weights_before, grid_weights, grid, cumulative_aggregate_time, cumulative_molecular_time = state #this is unnecessarily complicated but keeps the same structure as other methods
    system, kT, dt, nsegs, save_period, n_parallel, binbounds, bincenters = params

    #t3 = time.time()
    #run dynamics
    new_trjs, new_weights, new_weights_before, new_grid_weights, grid = propagators_v2.propagate_mtd(system, kT, trjs[-1].copy(), dt, nsegs, save_period, grid)
    trjs = np.concatenate((trjs, new_trjs), axis = 0)
    weights = np.concatenate((weights, new_weights), axis = 0)
    weights_before = np.concatenate((weights_before, new_weights_before), axis = 0)
    grid_weights = np.concatenate((grid_weights, new_grid_weights), axis = 0)
    #print("grid!")
    #print(grid_weights.shape)

    total_mtd_weights = np.sum(np.sqrt(np.multiply(weights, weights_before)), axis=1) #np.sum(weights_before, axis=1) + np.sum(weights, axis=1)

    weights_tiled = np.tile(total_mtd_weights, [n_parallel,1]).transpose()

    # t2 = time.time()
    # print(f"dynamics: {t2-t1}")

    #----------------------------------------------------------------------------------------------
    #populations associated with grid potential so we can see how well we've filled in the energy wells
    # pops_grid_uncorrected = np.exp(grid.grid)
    # pops_grid_uncorrected /= np.sum(pops_grid_uncorrected)

    #TODO move these to mtd_estimators.py
    #------------------------estimate populations from metadynamics grid-------------------------#

    pops_grid = np.exp(((kT+grid.dT)/grid.dT)*grid.grid)
    pops_grid /= np.sum(pops_grid)

    #----------------------------histogram-based population estimation----------------------------#

    #unlike np.digitize, np.histogram does not include data past the ends of the bin range, so we add artificial end bin boundaries
    binbounds_ends = [-999]+[bb for bb in binbounds]+[999]

    #estimate state populations from histogram
    #this will have to be replaced with a binning function that works for higher dimensions. 
    # It may make sense to abstract the binner object from WE into utility and use it here too
    pops_hist = np.histogram(trjs.flatten(), binbounds_ends, density=False)
    pops_hist = [ebp/len(trjs.flatten()) for ebp in pops_hist[0]]

    #-----------------------------estimate populations from grid in sampled bins only

    pops_grid_masked = [pg if h>0 else 0 for pg, h in zip(pops_grid, pops_hist)]
    pops_grid_masked /= np.sum(pops_grid_masked)

    #----------------------------combined grid+histogram-based population estimation----------------------------#
    #TODO is this dominated by weights of newer trajectories? Is the fact that these weights are not normalized a problem?
    pops_hist_weighted = np.histogram(trjs.flatten(), binbounds_ends, density=False, weights = weights.flatten())
    pops_hist_weighted = [ebp/np.sum(weights) for ebp in pops_hist_weighted[0]]

    # t3 = time.time()
    # print(f"histograms: {t3-t2}")

    ##################################################################################################
    #-------------------------------MSM-based population estimation----------------------------------#
    ##################################################################################################

    #assorted count reweighting schemes which did not work well
    #transition_weights = np.sqrt(np.divide(np.multiply(weights[1:].flatten(), weights_before[1:].flatten()), partition_functions))
    #transition_weights = np.sqrt(np.divide(weights[1:].flatten(), weights_before[1:].flatten()))
    #transition_weights = np.sqrt(weights[1:].flatten())

    #this count reweighting scheme seems to work the best but IDK why
    transition_weights = np.divide(np.sqrt(np.multiply(weights[1:].flatten(), weights_before[1:].flatten())), weights_tiled[1:].flatten())
    
    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 
    
    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    # and then reshaping to combine transitions from all parallel trajectories
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:]))
    transitions = transitions.reshape((2, transitions.shape[1]*transitions.shape[2])).transpose()

    #build MSM
    pops_msm = MSM_methods.transitions_to_eq_probs_v2(transitions, len(binbounds)+1, weights=transition_weights, silent=True, show_TPM=False)
    
    # #----------------------------MSM with nearby transitions only----------------------------#

    # #this does not work very well
    # transitions_nearby = np.array([tr for tr in transitions if abs(tr[0]-tr[1]) < 5])
    # transition_weights_nearby = np.array([w for tr, w in zip(transitions, transition_weights) if abs(tr[0]-tr[1]) < 5])
    # # print(transitions_nearby.shape)
    # # print(transition_weights_nearby.shape)

    # pops_msm_nearby = MSM_methods.transitions_to_eq_probs_v2(transitions_nearby, len(binbounds)+1, weights=transition_weights_nearby, show_TPM=False)

    # t4 = time.time()
    # print(f"MSM 1: {t4-t3}")

    ##################################################################################################
    #----------------------------MSM-based population estimation v2----------------------------------#
    ##################################################################################################

    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 

    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)

    n_states = len(binbounds)+1

    #all reweighted transition probability matrices, for averaging
    Pu_all = [] 

    #eigenvalues of each normalized Pu, for failed normalization scheme
    #eigenvalues = []

    #magnitude of entries in Pu over time; for debugging
    sum_vs_time = []

    #calculate reweighted TPMs for every round
    for t in range(len(trjs)-1):

        #count transitions
        transition_counts = np.zeros((n_states, n_states))
        for tr in transitions[t]:
            transition_counts[tr[1]][tr[0]] += 1
        
        #symmetrize matrix; does not have much effect
        #transition_counts_s = (transition_counts+transition_counts.transpose())/2

        #normalize transition count matrix to transition rate matrix
        #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
        # so that the probability associated with each element of X(t) is preserved 
        # (though not all at one index) when X(t) is multiplied by the TPM
        tpm = normalize(transition_counts, axis=0, norm='l1')

        #project vectors from unbiased basis into the basis with the metadynamics bias V
        #[:, np.newaxis] makes a row vector into a column vector
        reweight_matrix = np.outer(grid_weights[t], np.reciprocal(grid_weights[t]))

        Pu = np.multiply(reweight_matrix, tpm)

        #this does not improve performance since most weights are small and a few large ones dominate the average
        #eigenvalues.append(1/np.sum(Puv))
        #Puv /= np.sum(Puv)

        if t == len(trjs)-2 and False:
            print(t)
            plt.imshow(reweight_matrix, vmin = 0, vmax = 2)
            plt.title(f"rwm {t}")
            plt.show()
            
            print(f"total {np.sum(Pu)}")
            plt.plot(grid_weights[t])
            plt.show()

        Pu_all.append(Pu)

        sum_vs_time.append(np.sum(Pu))

    puv_arr = np.array(Pu_all)

    #VVVVVVVVVVVVVVVVVVVVVVVVVV THIS IS MATHEMATICALLY UNSOUND VVVVVVVVVVVVVVVVVVVVVVVVVV
    #TODO weight this mean by the number of counts in each column (different columns are weighted differently) 
    # to account for the different times at which the states were first sampled
    # if this is not done the later sampled states have lower mean transition "probabilities" overall which will affect the eigenvalue
    # when building a normal MSM this is not an issue because column normalization means that several rounds in which there are 
    # no transitions out of a state do not affect the probabilities from that state relative to other states (columns)
    # whereas when averaging here it lowers all the outgoing probabilities
    # and this can't be fixed by normalization because the matrix called Puv here (really Pu) is not supposed to be normalized
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    puv_mean = np.mean(puv_arr, axis=0)

    #calculate first eigenvector corresponding to equilibrium probability

    #for failed normalization scheme (see above)
    #eigenvalue = np.mean(eigenvalues)
    #print(f"eigenvalue: {eigenvalue}")
    #pops_msm_v2 = auxilliary_MSM_methods.tpm_2_eqprobs(puv_mean, eigenvalue)

    # pops_msm_v2 = MSM_methods.tpm_2_eqprobs(puv_mean, silent=True)
    # pops_msm_v2 /= np.sum(pops_msm_v2)
    pops_msm_v2 = np.zeros(len(puv_mean))

    t5 = time.time()
    # print(f"MSM 2: {t5-t4}")

    ##################################################################################################
    #----------------------------MSM-based population estimation v3----------------------------------#
    ##################################################################################################

    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 

    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)

    ##misnamed because I'm lazy
    #msm_v3_pops_all = mtd_estimators.MSM_v4(trjs, binbounds, grid_weights, system, bincenters, kT)
    msm_v3_pops_all = mtd_estimators.MSM_v3(transitions, binbounds, grid_weights, system, bincenters, kT)

    if False:
        ##################################################################################################
        #----------------------------MSM-based population estimation v3----------------------------------#
        ##################################################################################################

        #-----------------------------------------------------------------------------
        #outline

        #for every counts matrix c, find all other matrices c' which have sampled transitions arising from all of the same states (and possibly others),
        #   calculate equilibrium probabilities p_i from the subset of rows and columns in c' corresponding to the states from which transitions begin in c
        #   by averaging the reweighted TPMs calculated for the applicable rows and columns, 
        #   ergodically trimming the averaged reweighted TPM, and then calculating the eigenvector with eigenvalue 1
        #   [print these estimates for debugging]
        #for every pair of probability estimates p1 and p2, calculate the equilibrium probabilities for the set of states sampled by 
        #   both the counts matrices c1 and c2 used to calculate them, and calculate the associated energy difference: deltaG_12=-ktln(p1/p2)
        #   if there is no overlap, return 0 (some other data may be needed; see below)
        #calculate the overall energy shifts deltaG_i for each probability estimate which minimizes 
        #   the root mean square distance between between all overlapping pairs of energy estimates: sqrt(sum_i,j(deltaG_ij^2)))
        #   TODO find the solution to this analytically
        #shift the estimated landscape segments by deltaG_i and average all the estimates for each state k together to get the final landscape estimate

        #weighting based on how many times each transition (or each counts matrix) is reused is pending and may be unnecessary or intractable

        #-----------------------------------------------------------------------------
        #pseudocode

        # Cols_occupied(c) = np.where(np.sum(c, axis=1) > 0, 1, 0)    #aka cols
        # Submatrix(c, cols) = c[cols, cols]
        # Contains_cols(c’, c) = [sum(np.dot(cols(c), cols(c’))) == sum(cols(c))] #i.e. does c’ have transitions from all the same states as c?

        # #Subsets = [] # all matrices which have sampled all columns sampled in the c’th matrix

        #prob_est_all = []
        #cols_all = []

        # For i, c in enum C:
        #     cols = np.where(np.sum(c, axis=1) > 0, 1, 0)
        #     cols_all.append(cols)
        #     #Ssi = []
        #     mats_with_c_cols = []
        #     For j, c' in enum C:
        #         if contains_cols(c’, c):
        #             mats_with_c_cols.append(np.multiply(tpm(c'[cols, cols]), reweight_matrix[t][cols, cols]))
        #     #Subsets.append(ssi)
        #     prob_est = eig(mean(mats_with_c_cols)) # <-- TODO ergodic trimming here; at some point return separate estimates for all connected components instead of just the greatest one
        #     probs_est_allstates = {probs_est for states with estimates otherwise 0}
        #     prob_est_all.append(prob_est_allstates)

        #

        #-----------------------------------------------------------------------------
        #make transition count and reweighting matrices for each simulation round

        #this will have to be replaced with a binning function that works for higher dimensions
        trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 

        #calculate transitions by stacking the bin array with a time-shifted copy of itself
        transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)

        n_states = len(binbounds)+1
        nrounds = len(trjs)-1

        transition_counts_all = np.zeros((nrounds, n_states, n_states))
        reweight_matrices_all = []

        #there's probably some way to vectorize this but it's not currently (12/15/25) the limiting factor in code speed
        #calculate reweighted TPMs for every round
        for t in range(nrounds):
            #count transitions
            for tr in transitions[t]:
                transition_counts_all[t, tr[1], tr[0]] += 1

            reweight_matrices_all.append(np.outer(grid_weights[t], np.reciprocal(grid_weights[t])))

        reweight_matrices_all = np.stack(reweight_matrices_all)

        t6 = time.time()
        print(f"MSM 3, part 1: transitions and reweight matrices: {t6-t5}")

        #-----------------------------------------------------------------------------
        #calculate equilibrium probabilities for the set of states sampled in each round

        prob_est_all = []
        cols_all = []
        n_matrices = []

        #TODO; what if we just got the MSM free energy estimate for each pair of states (all n^2 of them) and then assembled those?
        #then we don't need to worry about finding pairs of simulations which sample exactly the same states
        #and ergodic trimming is rarely a problem and is very easy
        #this also lets weights account more easily for regional variation in sampling
        #and the number of matrices no longer keeps growing with the number of rounds

        for c1 in transition_counts_all:
            t11 = time.time()

            cols_1 = np.where(np.sum(c1, axis=1) > 0)[0]
            cols_1_1hot = np.where(np.sum(c1, axis=1) > 0, 1, 0)

            Pu_c_cols = []
            for c2, reweight_matrix in zip(transition_counts_all, reweight_matrices_all):
                #cols_2 = np.where(np.sum(c2, axis=1) > 0)[0]
                cols_2_1hot = np.where(np.sum(c2, axis=1) > 0, 1, 0)

                if np.dot(cols_1_1hot, cols_2_1hot) == sum(cols_1_1hot):

                    # print(c2)
                    # print(cols_1)
                    # print(c2[cols_1][:,cols_1])
                    # symmetrizing appears to make energy estimates worse, probably for reasons connected to nonequilibrium sampling or the asymmetry of the reweighting matrix
                    transition_counts_i = c2[cols_1][:,cols_1]
                    #transition_counts_i_s = (transition_counts_i+transition_counts_i.transpose())/2

                    #normalize transition count matrix to transition rate matrix
                    #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
                    # so that the probability associated with each element of X(t) is preserved 
                    # (though not all at one index) when X(t) is multiplied by the TPM
                    tpm = normalize(transition_counts_i, axis=0, norm='l1')

                    Pu_c_cols.append(np.multiply(tpm, reweight_matrix[cols_1][:,cols_1]))
            
            #for weighted averaging in the future?
            n_matrices.append(len(Pu_c_cols))

            Pu_arr = np.array(Pu_c_cols)
            Pu_mean = np.mean(Pu_arr, axis=0)

            t12 = time.time()
            #print(f"P_u matrix construction: {t12-t11}")

            #print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

            #TODO return separate estimates for all connected components instead of just the greatest one
            # failure to do this may have to do with slow convergence outside the main well

            #identify the greatest connected component of the transition counts matrix, 
            # which is only normally a problem when building haMSMs
            connected_states = connected_components(Pu_mean, directed=True, connection='strong')[1]
            cc_inds, cc_sizes = np.unique(connected_states, return_counts=True)

            if len(cc_sizes) == 0:
                print("no connected components detected")
                return [[1]], [0]

            greatest_connected_component = cc_inds[np.argmax(cc_sizes)]
                
            #remove all other components
            smaller_component_indices = [i for i, ccgroup in enumerate(connected_states) if ccgroup != greatest_connected_component]
            
            gcc_indices = np.where(connected_states == greatest_connected_component)[0]
            cols_1_connected = cols_1[gcc_indices]
            
            sc_indices = np.where(connected_states != greatest_connected_component)[0]
            cols_1_disconnected = cols_1[sc_indices]

            #print(sc_indices)
            #print(cols_1_1hot)
            
            cols_1_1hot[cols_1_disconnected] = 0
            #print(cols_1_1hot)

            cols_all.append(cols_1_1hot)

            # print(f"cols1: {cols_1}")
            # print(f"cols1_connected: {cols_1_connected}")
            # print(f"smaller comps: {smaller_component_indices}")
            
            Pu_mean = np.delete(Pu_mean, smaller_component_indices, 0)
            Pu_mean = np.delete(Pu_mean, smaller_component_indices, 1)

            # plt.imshow(Pu_mean)
            # plt.show()
            # plt.imshow(Pu_mean==0)
            # plt.show()

            #calculate equilibrium probabilities for the (connected) set of states sampled in c1
            pops_msm_i = auxilliary_MSM_methods.tpm_2_eqprobs_v3(Pu_mean)
            pops_msm_i /= np.sum(pops_msm_i) #I believe this is redundant

            #construct equilibrium probability vector for all states
            pops_msm_i_allstates = np.zeros(n_states)
            pops_msm_i_allstates[cols_1_connected] = pops_msm_i.flatten()

            for aaa, bbb in zip(pops_msm_i_allstates, cols_1_1hot):
                if (bbb == 0 and aaa != 0) or (bbb != 0 and aaa == 0):
                    print("error in reconstructing full-state probability vector")
                    print(pops_msm_i_allstates)
                    print(cols_1_1hot)
                    sys.exit(0)

            prob_est_all.append(pops_msm_i_allstates)

            t13 = time.time()
            #print(f"P_u matrix solution: {t13-t12}")

            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        t7 = time.time()
        print(f"MSM 3, part 2: MSM construction: {t7-t6}")

        #     plt.plot(pops_msm_i_allstates)
        # plt.show()

        # plt.plot(n_matrices)
        # plt.show()
        # import sys
        # sys.exit(0)

        #-----------------------------------------------------------------------------
        #align energy landscapes based on equal populations in overlapping states

        landscape_comparison_weights = np.zeros((len(prob_est_all), len(prob_est_all)))
        deltaG_matrix = np.zeros((len(prob_est_all), len(prob_est_all)))

        # print("-------------------------")

        for i, (p1, c1, n1) in enumerate(zip(prob_est_all, cols_all, n_matrices)):
            for j, (p2, c2, n2) in enumerate(zip(prob_est_all, cols_all, n_matrices)):
                c12 = np.multiply(c1,c2)
                if sum(c12) > 0 and i != j:
                    if sum(np.multiply(p2, c12)) == 0 or sum(np.multiply(p1, c12)) == 0:
                        print(f"p1: {np.multiply(p1, c12)}")
                        print(p1)
                        print(c1)
                        print(f"p2: {np.multiply(p2, c12)}")
                        print(p2)
                        print(c2)

                    pop_ratio = sum(np.multiply(p2, c12))/sum(np.multiply(p1, c12))
                    deltaG_matrix[i,j] = -kT*np.log(pop_ratio)
                    #I have no theoretical justification for this specific weighting scheme
                    landscape_comparison_weights[i,j] = n1*n2*sum(c12) #weight by number of matrices used to calculate each estimate and number of overlapping states
                    #not including n1 and n2 made things worse anecdotally

        # print("-------------------------")

        t8 = time.time()
        print(f"MSM 3, part 3: landscape comparison matrix: {t8-t7}")

        fe_shifts = align_free_energy_offsets(deltaG_matrix, landscape_comparison_weights, gauge="zero")
        #minimize_weighted_G2(deltaG_matrix, landscape_comparison_weights) #method written by chatGPT
        # print(fe_shifts)
        # plt.hist(fe_shifts)
        # plt.show()

        #Beware this is full of infinities for unsampled states
        with np.errstate(divide = 'ignore'):
            deltaG_shifted = np.stack([-kT*np.log(prob_est/kT)-shift for prob_est, shift in zip(prob_est_all, fe_shifts)])
        #print(deltaG_shifted)
        #plt.matshow(deltaG_shifted)  # <-- very useful
        #plt.show()

        cols_mat = np.stack(cols_all)

        states_sampled = np.where(np.sum(cols_mat, axis=0) > 0)[0]
        states_sampled_1hot = np.where(np.sum(cols_mat, axis=0) > 0, 1, 0)

        #print(cols_mat.shape)

        n_matrices_tiled = np.tile(np.array(n_matrices), (n_states,1)).transpose()
        #print(n_matrices_tiled)
        #print(np.tile(np.array(n_matrices), (cols_mat.shape[1],1)).transpose().shape)
        weight_matrix = cols_mat #np.multiply(cols_mat, n_matrices_tiled)

        deltaG_average_sampled = np.average(np.nan_to_num(deltaG_shifted[:,states_sampled], posinf=0.0, neginf=0.0), axis=0, weights=weight_matrix[:,states_sampled])

        msm_v3_pops_sampled = np.exp(-deltaG_average_sampled/kT)
        msm_v3_pops_sampled /= np.sum(msm_v3_pops_sampled)
        msm_v3_pops_all = np.zeros(n_states)
        msm_v3_pops_all[states_sampled] = msm_v3_pops_sampled

        deltaG_average = np.zeros(n_states)
        deltaG_average[states_sampled] = deltaG_average_sampled

        t9 = time.time()
        print(f"MSM 3, part 4: final assembly: {t9-t8}")

        #print(deltaG_average)

        for gi in deltaG_shifted:
            plt.plot(gi)
        
        plt.plot(states_sampled, deltaG_average_sampled, linewidth=2, color='black', zorder = 999999)
        
        plt.plot(deltaG_shifted[-1], linewidth=2, color='grey', linestyle="dashed", zorder = 99999)

        true_populations, true_energies = system.normalized_pops_energies(kT, bincenters)

        plt.plot(-kT*np.log(true_populations), linewidth=2, color='red', zorder = 99998)

        plt.show()

    


    #----------------------------------increment simulation time--------------------------------

    cumulative_molecular_time += nsegs*save_period
    cumulative_aggregate_time += n_parallel*nsegs*save_period

    return (trjs, weights, weights_before, grid_weights, grid, cumulative_aggregate_time, cumulative_molecular_time), (cumulative_aggregate_time, cumulative_molecular_time, pops_hist_weighted, pops_msm, msm_v3_pops_all), False #pops_grid_uncorrected, pops_hist, pops_grid, pops_msm_v2, pops_grid_masked, 



#set up and run parallel simulations and estimate the energy landscape with a histogram
def sampler_parallel_hist_mtd(system_args, resource_args, bin_args, sampler_params):

    #----------------------------------input handling--------------------------------

    dT, stdev, rate_per_frame = sampler_params

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

    #adjust gaussian height so that potential energy is added at a constant rate 
    # regardless of the number of parallel simulations or gaussian deposition frequency
    rate = rate_per_frame*save_period/n_parallel

    print("\n")
    print("---------------------MULTIPLE WALKER METADYNAMICS---------------------")
    print(f"running {n_parallel} parallel multiple walker metadynamics simulations")
    print(f"molecular time: {actual_molecular_time} steps;  aggregate time: {actual_aggregate_time} steps")
    print(f"data points saved: {nsegs*n_timepoints*n_parallel} at {save_period}-step intervals")
    print(f"gaussians of height {rate} are added every {save_period} steps")

    # #determine number of parallel simulations and steps per simulation
    # n_parallel = int(round(aggregate_simulation_limit/molecular_time_limit))
    # nsteps = int(round(aggregate_simulation_limit/(n_parallel*n_timepoints)))

    # print("\n")
    # print("---------------------MULTIPLE WALKER METADYNAMICS---------------------")
    # print(f"running {n_parallel} parallel multiple walker metadynamics simulations for {nsteps*n_timepoints} steps each")
    # print(f"molecular time: {nsteps*n_timepoints} steps;  aggregate time: {nsteps*n_timepoints*n_parallel} steps")
    # print(f"data points saved: {aggregate_simulation_limit/save_period} at {save_period}-step intervals")
    # print(f"gaussians of height {rate} are added every {save_period} steps")

    #--------------------------------set up and run system-----------------------------

    #initiate all simulations in the same state
    trjs = np.array([system.standard_init_coord for element in range(n_parallel)]).reshape((1, n_parallel, len(system.standard_init_coord)))
    init_weights = np.repeat(1, n_parallel).reshape((1, n_parallel)) #surely there is a better way to do this with tile or something
    init_weights_before = np.repeat(1, n_parallel).reshape((1, n_parallel))

    grid = metadynamics_v1.grid(system.standard_analysis_range, len(binbounds)-1, rate, dT, stdev)

    init_grid_weights = grid.grid_weights(kT).reshape((1, len(binbounds)+1))

    #pack the initial state and parameters and run dynamics
    initial_state = (trjs, init_weights, init_weights_before, init_grid_weights, grid, 0, 0)
    params = (system, kT, dt, nsegs, save_period, n_parallel, binbounds, bincenters)
    time_x_observables = utility_v1.run_for_n_timepoints(parallel_trj_histogram_mtd, params, initial_state, n_timepoints)

    #effectively transpose the list of lists so the first axis is observable type rather than time
    #but without the data type/structure requirement of a numpy array
    observables_x_time = [list(row) for row in zip(*time_x_observables)]

    observable_names = ["MWM: weighted histogram", "MWM: count-reweighted MSM", "MWM: franken MSM"]

    return observables_x_time, observable_names





























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




    # trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 
    # #print(trjs_ditigized.shape)

    # #calculate transitions by stacking the bin array with a time-shifted copy of itself
    # transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)
    # #print(transitions.shape)

    # n_states = len(binbounds)+1

    # #all reweighted transition probability matrices, for averaging
    # Puv_all = [] #TODO: rename; this is misnamed relative to my notebook contents

    # eigenvalues = []

    # sum_vs_time = []

    # #calculate reweighted TPMs for every round
    # for t in range(len(trjs)-1):

    #     #count transitions
    #     transition_counts = np.zeros((n_states, n_states))
    #     for tr in transitions[t]:
    #         transition_counts[tr[1]][tr[0]] += 1
        
    #     #symmetrize matrix
    #     #transition_counts_s = (transition_counts+transition_counts.transpose())/2

    #     # plt.imshow(transition_counts)
    #     # plt.title(f"tcm {t}")
    #     # plt.show()
    #     #normalize transition count matrix to transition rate matrix
    #     #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
    #     # so that the probability associated with each element of X(t) is preserved 
    #     # (though not all at one index) when X(t) is multiplied by the TPM
    #     tpm = normalize(transition_counts, axis=0, norm='l1')
    #     # plt.imshow(tpm)
    #     # plt.title(f"tpm {t}")
    #     # plt.show()

    #     # print(np.reciprocal(grid_weights[t]))
    #     # print(grid_weights[t][:, np.newaxis])

    #     #[:, np.newaxis] makes a row vector into a column vector
    #     reweight_matrix = np.outer(grid_weights[t], np.reciprocal(grid_weights[t]))

    #     # if t%100 == 0:


    #     #print(reweight_matrix.shape)
    #     Puv = np.multiply(reweight_matrix, tpm)
    #     #the normalization below should work if sum(1/Puv_t)/n is used as the eigenvalue instead of 1
        
    #     # print("-------------------------------------")
    #     # plt.imshow(Puv, vmin = 0, vmax = 1)
    #     # plt.show()

    #     #this does not improve performance since most weights are small and a few large ones dominate the average
    #     #eigenvalues.append(1/np.sum(Puv))
    #     #Puv /= np.sum(Puv)

    #     # plt.imshow(Puv)
    #     # plt.show()

    #     if t == len(trjs)-2 and False:
    #         print(t)
    #         plt.imshow(reweight_matrix, vmin = 0, vmax = 2)
    #         plt.title(f"rwm {t}")
    #         plt.show()
            
    #         print(f"total {np.sum(Puv)}")
    #         plt.plot(grid_weights[t])
    #         plt.show()

    #     Puv_all.append(Puv)

    #     sum_vs_time.append(np.sum(Puv))

    # puv_arr = np.array(Puv_all)
    # #print(puv_arr.shape)

    # #VVVVVVVVVVVVVVVVVVVVVVVVVV THIS IS MATHEMATICALLY UNSOUND VVVVVVVVVVVVVVVVVVVVVVVVVV
    # #TODO weight this mean by the number of counts in each column (different columns are weighted differently) 
    # # to account for the different times at which the states were first sampled
    # # if this is not done the later sampled states have lower mean transition "probabilities" overall which will affect the eigenvalue
    # # when building a normal MSM this is not an issue because column normalization means that several rounds in which there are 
    # # no transitions out of a state do not affect the probabilities from that state relative to other states (columns)
    # # whereas when averaging here it lowers all the outgoing probabilities
    # # and this can't be fixed by normalization because the matrix called Puv here (really Pu) is not supposed to be normalized
    # puv_mean = np.mean(puv_arr, axis=0)

    # #eigenvalue = np.mean(eigenvalues)

    # #print(f"eigenvalue: {eigenvalue}")

    # # plt.imshow(puv_mean)
    # # plt.show()

    # # plt.plot(sum_vs_time)
    # # plt.show()

    # #pops_msm_v2 = auxilliary_MSM_methods.tpm_2_eqprobs(puv_mean, eigenvalue)
    # pops_msm_v2 = MSM_methods.tpm_2_eqprobs(puv_mean)
    # pops_msm_v2 /= np.sum(pops_msm_v2)


    # #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


