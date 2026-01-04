import time
import sys
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
from numpy.linalg import solve

import MSM_methods
import auxilliary_MSM_methods


# by chatGPT
def align_free_energy_offsets(G, W, gauge="mean"):
    """
    Align relative free-energy estimates G_ij ≈ v_j - v_i

    Parameters
    ----------
    G : (n, n) ndarray
        Pairwise free energy estimates (NaN or arbitrary where no overlap)
    W : (n, n) ndarray
        Weights (0 where no overlap)
    gauge : str
        'mean' -> sum(v)=0
        'zero' -> v[0]=0

    Returns
    -------
    v : (n,) ndarray
        Optimal free-energy offsets
    """
    G = np.asarray(G, float)
    W = np.asarray(W, float)
    n = G.shape[0]

    # Laplacian
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = W[i, j] + W[j, i]
            if w > 0:
                L[i, i] += w
                L[j, j] += w
                L[i, j] -= w
                L[j, i] -= w

    # plt.imshow(L)
    # plt.show()

    # RHS
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.sum(W[:, i] * G[:, i]) - np.sum(W[i, :] * G[i, :])

    # Fix gauge
    if gauge == "mean":
        L[-1, :] = 1.0
        b[-1] = 0.0
    elif gauge == "zero":
        L[0, :] = 0.0
        L[0, 0] = 1.0
        b[0] = 0.0
    else:
        raise ValueError("Unknown gauge")

    return solve(L, b)



##################################################################################################
#----------------------------MSM-based population estimation v3----------------------------------#
##################################################################################################

def MSM_v4(trjs, binbounds, grid_weights, system, bincenters, kT):

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
    t5 = time.time()

    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 

    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)

    n_states = len(binbounds)+1
    nrounds = len(trjs)-1

    transition_counts_all = np.zeros((nrounds, n_states, n_states))
    #tpm_all = np.zeros((nrounds, n_states, n_states))
    reweight_matrices_all = np.zeros((nrounds, n_states, n_states))

    #there's probably some way to vectorize this but it's not currently (12/15/25) the limiting factor in code speed
    #calculate reweighted TPMs for every round
    for t in range(nrounds):
        #count transitions
        for tr in transitions[t]:
            transition_counts_all[t, tr[1], tr[0]] += 1

        reweight_matrices_all[t] = np.outer(grid_weights[t], np.reciprocal(grid_weights[t]))

        #tpm_all[t] = normalize(transition_counts_all[t], axis=1, norm='l1') #TODO can't do this in here since the normalization is different with only 2 states

    #p_u = np.multiply(tpm_all, reweight_matrices_all)

    #q = np.multiply

    t6 = time.time()
    print(f"MSM 4, part 1: transitions and reweight matrices: {t6-t5}")

    #-----------------------------------------------------------------------------
    #estimate relative free energy of each pair of states
    n_pairs = int(round(n_states*(n_states-1)/2))

    probs_est = np.zeros([n_pairs, 2])
    state_pairs = np.zeros([n_pairs, 2], dtype=int)
    interstate_transitions = np.zeros(n_pairs)
    used_transitions = np.zeros(n_pairs)

    dG = []
    dG_mat = np.zeros((n_states, n_states))

    #t_norm = 0

    pair_i = 0
    for s1 in range(n_states):
        for s2 in range(s1+1, n_states):
            state_pairs[pair_i] = [s1, s2]

            #take the submatrix of p_u corresponding to transitions between states s1 and s2
            counts_12 = transition_counts_all[:,(s1,s2)][:,:,(s1,s2)] #can this be reduced to one indexing operation?
            #counts_12 = (counts_12.transpose(0,2,1)+counts_12.transpose(0,1,2))/2 #symmetrize transition counts    <--- this makes the results much worse
            #print(counts_12.shape)

            reweight_12 = reweight_matrices_all[:,(s1,s2)][:,:,(s1,s2)] #can this be reduced to one indexing operation?

            both_sampled = np.where(np.prod(np.sum(counts_12, axis=1), axis=1)>0)
            #print(both_sampled)

            counts_12_sampled = counts_12[both_sampled]
            reweight_12_sampled = reweight_12[both_sampled]

            #ta = time.time()
            #normalize transition count matrix to transition rate matrix
            col_sums = np.sum(counts_12_sampled, axis=1, keepdims=True)
            p_uv = counts_12_sampled / col_sums

            # p_uv = np.zeros(counts_12.shape)
            # for t in range(nrounds):
            #     p_uv[t] = normalize(counts_12[t], axis=0, norm='l1')
            #tb = time.time()

            #t_norm += (tb-ta)

            #reweight transition probability matrix
            p_u = np.multiply(p_uv, reweight_12_sampled)


            #q_ind = {1 if transitions begin from both states at each time point, 0 otherwise}
            #this is to avoid native-state bias from including transition matrices which don't sample both states
            both_sampled_1hot = np.where(np.prod(np.sum(p_u, axis=1), axis=1)>0, 1, 0)
            #both_sampled = np.where(np.multiply(counts_12[:,0,1], counts_12[:,1,0])>0, 1, 0)

            #total number of transitions at each time point, for weighting
            n_transitions_t = np.sum(counts_12_sampled, axis=(1,2))
            # print(n_transitions_t)
            # print(n_transitions_t.shape)

            if sum(both_sampled_1hot) > 0: #if there are no transitions at all (i.e. both states are unsampled), continue to the next state pair (averaging will crash)

                pair_weights_t = np.multiply(both_sampled_1hot, n_transitions_t)

                p_u_avg = np.average(p_u, axis=0, weights=pair_weights_t)

                #ergodic trimming, which is very easy for two states
                if p_u_avg[0,1] > 0 and p_u_avg[1,0] > 0: # and p_u_avg[0,0] > 0 and p_u_avg[1,1] > 0:
                    # print(pair_weights_t)
                    # print(p_u_avg)
                    #I think this would create biases if used for averaging across time slices but not for weighting between different pairs of states
                    interstate_transitions[pair_i] = np.sum((counts_12_sampled[:,0,1], counts_12_sampled[:,1,0]))
                    used_transitions[pair_i] = np.sum(pair_weights_t)
                    #larger eigenvalue calculated with quadratic formula
                    l_0 = (p_u_avg[0,0]+p_u_avg[1,1] + np.sqrt((p_u_avg[0,0]+p_u_avg[1,1])**2 - 4*(p_u_avg[0,0]*p_u_avg[1,1] - p_u_avg[0,1]*p_u_avg[1,0])))/2
                    #print(l_0)

                    #print(f"discriminant = {(p_u_avg[0,0] + p_u_avg[1,1])**2 - 4*(p_u_avg[0,0]*p_u_avg[1,1] - p_u_avg[0,1]*p_u_avg[1,0])}")

                    #calculate normalized eigenvector and add it to probs_est
                    #l_0 = 1 #this is always true for individual 2x2 rewighted TPMs
                    x1 = -p_u_avg[0,1]/(p_u_avg[0,0]-l_0-p_u_avg[0,1])
                    x2 = 1-x1

                    if x1 > 1 or x1 < 0:
                        print(f"eigenvector calculation error: x1={x1}, x2={x2}")
                        print(p_u_avg)
                        sys.exit(0)

                    probs_est[pair_i] = [x1, x2]

                    dG.append(-kT*np.log(x2/x1))
                    dG_mat[s1,s2] = -kT*np.log(x2/x1)

            pair_i += 1

    # plt.hist([dGi for dGi in dG if dGi != 0], bins=50)
    # plt.show()


    # # display_mat = np.zeros((n_states, n_states))
    # # for sp, dg in zip(state_pairs, dG):
    # #     print(sp)
    # #     display_mat[sp[0], sp[1]] = dg
    # plt.imshow(dG_mat)
    # plt.colorbar()
    # plt.show()


    t7 = time.time()
    print(f"MSM 4, part 2: pairwise population estimates: {t7-t6}")
    #print(f"normalization time: {t_norm}")



    #-----------------------------------------------------------------------------
    #calculate equilibrium probabilities for the set of states sampled in each round
    if False:
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
        print(f"MSM 4, part 2: MSM construction: {t7-t6}")

    #     plt.plot(pops_msm_i_allstates)
    # plt.show()

    # plt.plot(n_matrices)
    # plt.show()
    # import sys
    # sys.exit(0)

    #-----------------------------------------------------------------------------
    #align energy landscapes based on equal populations in overlapping states

    landscape_comparison_weights = np.zeros((n_pairs, n_pairs))
    deltaG_matrix = np.zeros((n_pairs, n_pairs))

    #ohe hot encodings of which states are sampled in each pairwise estimate
    cols_mat = np.zeros((n_pairs, n_states))
    #probability estimates for all states using only each pair. 
    #These are meaningless on their own (since all but the two sampled states have 0 probability) but useful for final assembly
    probs_est_allstates = np.zeros((n_pairs, n_states))

    # print("-------------------------")

    for i, (p1, c1, n1) in enumerate(zip(probs_est, state_pairs, interstate_transitions)):
        probs_est_allstates[i, c1[0]] = p1[0]
        probs_est_allstates[i, c1[1]] = p1[1]
        if n1 > 0:
            cols_mat[i, c1[0]] = 1
            cols_mat[i, c1[1]] = 1

        for j, (p2, c2, n2) in enumerate(zip(probs_est, state_pairs, interstate_transitions)):
            if n1 == 0 or n2 == 0:
                continue 
            #strictly upper triangular matrices cause problems because they are singular
            ##not true --> we only need the upper triangular part of the matrix; the other half is redundant

            #this is surely not the most elegant way to find the population ratio of overlapping states
            if c1[0] == c2[0]:
                deltaG_matrix[i,j] = -kT*np.log(p2[0]/p1[0])
            elif c1[0] == c2[1]:
                deltaG_matrix[i,j] = -kT*np.log(p2[1]/p1[0])
            elif c1[1] == c2[0]:
                deltaG_matrix[i,j] = -kT*np.log(p2[0]/p1[1])
            elif c1[1] == c2[1]:
                deltaG_matrix[i,j] = -kT*np.log(p2[1]/p1[1])
            else:
                continue #skip adding a weight if there is no overlap

            #prioritize the alignment of states with energy estimates built on more data
            #I don't have a theoretical justification for this specific weighting function
            landscape_comparison_weights[i,j] = np.sqrt(n1*n2)

            # c12 = np.multiply(c1,c2)
            # if sum(c12) > 0 and i != j:
            #     if sum(np.multiply(p2, c12)) == 0 or sum(np.multiply(p1, c12)) == 0:
            #         print(f"p1: {np.multiply(p1, c12)}")
            #         print(p1)
            #         print(c1)
            #         print(f"p2: {np.multiply(p2, c12)}")
            #         print(p2)
            #         print(c2)

            #     pop_ratio = sum(np.multiply(p2, c12))/sum(np.multiply(p1, c12))
            #     deltaG_matrix[i,j] = -kT*np.log(pop_ratio)
            #     #I have no theoretical justification for this specific weighting scheme
            #     landscape_comparison_weights[i,j] = n1*n2 #weight by number of matrices used to calculate each estimate and number of overlapping states

    pairs_estimated = np.where(np.sum(landscape_comparison_weights, axis=0) > 0)[0]
    lcw_est = landscape_comparison_weights[pairs_estimated][:, pairs_estimated]
    dg_est = deltaG_matrix[pairs_estimated][:, pairs_estimated]
    # print(states_estimated)
    # print(deltaG_matrix[states_estimated][:, states_estimated])
    # print(deltaG_matrix[states_estimated][:, states_estimated].shape)

    # print(np.max(landscape_comparison_weights))
    # lcw_est_masked = np.ma.masked_where(lcw_est == 0, lcw_est)
    # plt.imshow(lcw_est_masked)#, vmin = 0, vmax=1)
    # plt.show()
    # # print(np.max(deltaG_matrix))

    # dg_est_masked = np.ma.masked_where(lcw_est == 0, dg_est)
    # plt.imshow(dg_est_masked)#, vmin = 0, vmax=1)
    # plt.show()

    # # print(deltaG_matrix)
    # deltaG_nonzero = [dGij for dGij in deltaG_matrix.flatten() if dGij != 0]
    # plt.hist(deltaG_nonzero)
    # plt.show()

    # print("-------------------------")

    t8 = time.time()
    print(f"MSM 4, part 3: landscape comparison matrix: {t8-t7}")

    # print(dg_est.shape)
    # print(lcw_est.shape)

    #connectivity trimming
    #ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    connected_states = connected_components(dg_est, directed=True, connection='strong')[1]

    cc_inds, cc_sizes = np.unique(connected_states, return_counts=True)

    if len(cc_sizes) == 0:
        print("no connected components detected")
        return [[1]], [0]

    greatest_connected_component = cc_inds[np.argmax(cc_sizes)]
    
    smaller_component_indices = [i for i, ccgroup in enumerate(connected_states) if ccgroup != greatest_connected_component]

    dg_est  = np.delete(dg_est,  smaller_component_indices, 0)
    dg_est  = np.delete(dg_est,  smaller_component_indices, 1)
    lcw_est = np.delete(lcw_est, smaller_component_indices, 0)
    lcw_est = np.delete(lcw_est, smaller_component_indices, 1)

    gcc_indices = np.where(connected_states == greatest_connected_component)[0]
    pairs_estimated = pairs_estimated[gcc_indices]

    # lcw_est_masked = np.ma.masked_where(lcw_est == 0, lcw_est)
    # plt.imshow(lcw_est_masked)#, vmin = 0, vmax=1)
    # plt.show()

    # dg_est_masked = np.ma.masked_where(lcw_est == 0, dg_est)
    # plt.imshow(dg_est_masked)#, vmin = 0, vmax=1)
    # plt.show()

    #ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

    fe_shifts = align_free_energy_offsets(dg_est, lcw_est, gauge="zero") #function written by chatGPT

    fe_shifts_allstates = np.zeros(n_pairs)
    fe_shifts_allstates[pairs_estimated] = fe_shifts

    #fe_shifts = align_free_energy_offsets(deltaG_matrix, landscape_comparison_weights, gauge="zero")
    #minimize_weighted_G2(deltaG_matrix, landscape_comparison_weights) #method written by chatGPT
    # print(fe_shifts)
    # plt.hist(fe_shifts)
    # plt.show()

    #Beware this is full of infinities for unsampled states
    with np.errstate(divide = 'ignore'):
        deltaG_shifted = np.stack([-kT*np.log(prob_est/kT)-shift for prob_est, shift in zip(probs_est_allstates, fe_shifts_allstates)])
    #print(deltaG_shifted.shape)
    #plt.matshow(deltaG_shifted)  # <-- very useful
    #plt.show()

    #cols_mat = np.stack(cols_all)

    states_sampled = np.unique(state_pairs[pairs_estimated].flatten())
    #states_sampled = np.where(np.sum(cols_mat, axis=0) > 0)[0]
    #states_sampled_1hot = np.where(np.sum(cols_mat, axis=0) > 0, 1, 0)

    #print(cols_mat.shape)

    #TODO use for weighting < this doesn't seem to help
    n_matrices_tiled = np.tile(used_transitions, (n_states,1)).transpose()
    # print(interstate_transitions)
    # plt.plot(interstate_transitions)
    # plt.show()
    # print(n_matrices_tiled.shape)
    # print(cols_mat.shape)
    # #plt.imshow(n_matrices_tiled)
    # plt.show()

    #print(np.tile(np.array(n_matrices), (cols_mat.shape[1],1)).transpose().shape)

    #weighting seems to make things worse
    #weight_matrix = cols_mat 
    weight_matrix = np.multiply(cols_mat, n_matrices_tiled)
    #print(cols_mat.shape)
    # plt.plot(weight_matrix)
    # plt.show()

    deltaG_average_sampled = np.average(np.nan_to_num(deltaG_shifted[:,states_sampled], posinf=0.0, neginf=0.0), axis=0, weights=weight_matrix[:,states_sampled])

    msm_pops_sampled = np.exp(-deltaG_average_sampled/kT)
    z = np.sum(msm_pops_sampled)
    msm_pops_sampled /= np.sum(msm_pops_sampled)
    msm_pops_all = np.zeros(n_states)
    msm_pops_all[states_sampled] = msm_pops_sampled

    #deltaG_average = np.zeros(n_states)
    #deltaG_average[states_sampled] = -kT*np.log(msm_pops_sampled) #deltaG_average_sampled

    t9 = time.time()
    print(f"MSM 4, part 4: final assembly: {t9-t8}")

    #print(deltaG_average)

    for gi in deltaG_shifted:
        plt.plot(gi)
    
    plt.plot(states_sampled, deltaG_average_sampled+kT*np.log(z), linewidth=2, color='black', zorder = 999999)
    
    plt.plot(deltaG_shifted[-1], linewidth=2, color='grey', linestyle="dashed", zorder = 99999)

    true_populations, true_energies = system.normalized_pops_energies(kT, bincenters)

    plt.plot(-kT*np.log(true_populations), linewidth=2, color='red', zorder = 99998)

    plt.show()

    return msm_pops_all




##################################################################################################
#----------------------------MSM-based population estimation v3----------------------------------#
##################################################################################################

def MSM_v3(trjs, binbounds, grid_weights, system, bincenters, kT):

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
    t5 = time.time()

    #this will have to be replaced with a binning function that works for higher dimensions
    trjs_ditigized = np.digitize(trjs, binbounds).reshape((trjs.shape[0], trjs.shape[1])) 

    #calculate transitions by stacking the bin array with a time-shifted copy of itself
    transitions = np.stack((trjs_ditigized[:-1], trjs_ditigized[1:])).transpose(1,2,0)

    n_states = len(binbounds)+1
    nrounds = len(trjs)-1

    transition_counts_all = np.zeros((nrounds, n_states, n_states))
    reweight_matrices_all = np.zeros((nrounds, n_states, n_states))

    #there's probably some way to vectorize this but it's not currently (12/15/25) the limiting factor in code speed
    #calculate reweighted TPMs for every round
    for t in range(nrounds):
        #count transitions
        for tr in transitions[t]:
            transition_counts_all[t, tr[1], tr[0]] += 1

        reweight_matrices_all[t] = np.outer(grid_weights[t], np.reciprocal(grid_weights[t]))

    t6 = time.time()
    print(f"MSM 3, part 1: transitions and reweight matrices: {t6-t5}")

    #-----------------------------------------------------------------------------
    #calculate equilibrium probabilities for the set of states sampled in each round

    prob_est_all = []
    cols_all_connected = []
    n_matrices = []

    #cols_all      = np.where(np.sum(transition_counts_all, axis=1) > 0)
    #cols_all_1hot = np.where(np.sum(transition_counts_all, axis=1) > 0, 1, 0)


    #TODO; what if we just got the MSM free energy estimate for each pair of states (all n^2 of them) and then assembled those?
    #then we don't need to worry about finding pairs of simulations which sample exactly the same states
    #and ergodic trimming is rarely a problem and is very easy
    #this also lets weights account more easily for regional variation in sampling
    #and the number of matrices no longer keeps growing with the number of rounds

    inner_loop_time = 0

    for ci, c1 in enumerate(transition_counts_all):
        t11 = time.time()

        # print("-------------")
        # print(np.where(np.sum(c1, axis=0) > 0)[0])
        # print(np.where(np.sum(c1, axis=1) > 0)[0])

        cols_1 = np.where(np.sum(c1, axis=0) > 0)[0]          #axis??
        cols_1_1hot = np.where(np.sum(c1, axis=0) > 0, 1, 0)          #axis?? #cols_all_1hot[ci] #

        #print(cols_1)

        t6a = time.time()

        #specifically timesteps where transitions from all states in cols_1 to other states in cols_1 are sampled
        cols_i_1hot_nonbin = np.prod(np.sum(transition_counts_all[:,cols_1][:,:,cols_1], axis=1), axis = 1) #not a true 1 hot encoding since contents can be >1
        timesteps_with_matching_cols = np.where(cols_i_1hot_nonbin > 0)[0]
        
        if len(timesteps_with_matching_cols) == 0:
            #print("no matching columns found")
            #print(timesteps_with_matching_cols)
            continue
        #timesteps_with_matching_cols = np.where(cols_all_1hot @ cols_1_1hot == sum(cols_1_1hot))[0]
        # print(timesteps_with_matching_cols)

        # print(transition_counts_all.shape)
        # print(transition_counts_all[timesteps_with_matching_cols].shape)
        # print(transition_counts_all[timesteps_with_matching_cols][:,cols_1].shape)

        transition_counts_i = transition_counts_all[timesteps_with_matching_cols][:,cols_1][:,:,cols_1]
        #print(transition_counts_i.shape)

        #normalize transition count matrix to transition rate matrix
        col_sums = np.sum(transition_counts_i, axis=1, keepdims=True)
        #print(col_sums.shape)

        #if all the transitions from a column go outside the set of columns for which transitions were sampled, skip this round
        #in the future do ergodic trimming instead


        # plt.imshow(col_sums)
        # plt.show()
        P_uv = transition_counts_i / col_sums
        #print(P_uv.shape)
        reweight_matrices_i = reweight_matrices_all[timesteps_with_matching_cols][:,cols_1][:,:,cols_1]
        Puv_reweighted = np.multiply(P_uv, reweight_matrices_i)
        Puv_reweighted_avg = np.mean(Puv_reweighted, axis=0)

        if False: #slower but more readable code for constructing Pu_c_cols
            Pu_c_cols = []
            for c2, reweight_matrix in zip(transition_counts_all, reweight_matrices_all):
                #cols_2 = np.where(np.sum(c2, axis=1) > 0)[0]
                cols_2_1hot = np.where(np.sum(c2, axis=0) > 0, 1, 0)          #axis??

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

        t6b = time.time()
        inner_loop_time += t6b - t6a

        n_matrices.append(len(timesteps_with_matching_cols))

        #for weighted averaging in the future?
        #n_matrices.append(len(Pu_c_cols))

        #Pu_arr = np.array(Pu_c_cols)
        #Puv_reweighted_avg = np.mean(Pu_arr, axis=0)

        t12 = time.time()
        #print(f"P_u matrix construction: {t12-t11}")

        #print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")

        #TODO return separate estimates for all connected components instead of just the greatest one
        # failure to do this may have to do with slow convergence outside the main well

        #identify the greatest connected component of the transition counts matrix, 
        # which is only normally a problem when building haMSMs
        connected_states = connected_components(Puv_reweighted_avg, directed=True, connection='strong')[1]
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

        cols_all_connected.append(cols_1_1hot)

        # print(f"cols1: {cols_1}")
        # print(f"cols1_connected: {cols_1_connected}")
        # print(f"smaller comps: {smaller_component_indices}")
        
        Puv_reweighted_avg = np.delete(Puv_reweighted_avg, smaller_component_indices, 0)
        Puv_reweighted_avg = np.delete(Puv_reweighted_avg, smaller_component_indices, 1)

        # plt.imshow(Pu_mean)
        # plt.show()
        # plt.imshow(Pu_mean==0)
        # plt.show()

        #calculate equilibrium probabilities for the (connected) set of states sampled in c1
        pops_msm_i = auxilliary_MSM_methods.tpm_2_eqprobs_v3(Puv_reweighted_avg)
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

    print(f"MSM 3, part 2: MSM construction inner loop: {inner_loop_time}")


    #     plt.plot(pops_msm_i_allstates)
    # plt.show()

    # plt.plot(n_matrices)
    # plt.show()
    # import sys
    # sys.exit(0)

    #-----------------------------------------------------------------------------
    #align energy landscapes based on equal populations in overlapping states

    #landscape_comparison_weights = np.zeros((len(prob_est_all), len(prob_est_all)))
    #deltaG_matrix = np.zeros((len(prob_est_all), len(prob_est_all)))

    print("-------------------------")
    c1_mat = np.tile(np.stack(cols_all_connected),(len(cols_all_connected),1,1))
    c2_mat = c1_mat.transpose((1,0,2))
    c1c2 = np.multiply(c1_mat, c2_mat) #1-hot-encoded overlap vectors between each pair of estimates, forming a 3d [n_estimates, n_estimates, n_states] array
    overlap_areas = np.sum(c1c2, axis=2)

    landscape_comparison_weights = np.multiply(np.outer(n_matrices, n_matrices), overlap_areas)

    p1_mat = np.tile(np.stack(prob_est_all),(len(prob_est_all),1,1))
    p2_mat = p1_mat.transpose((1,0,2))

    pc1_mat = np.sum(np.multiply(p1_mat, c1_mat), axis=2)
    pc2_mat = np.sum(np.multiply(p2_mat, c1_mat), axis=2)
    print(pc1_mat.shape)

    deltaG_matrix = np.divide(pc2_mat, pc1_mat, out=np.zeros_like(pc2_mat), where=pc1_mat!=0)

    plt.imshow(deltaG_matrix)
    plt.show()

    if False:
        for i, (p1, c1, n1) in enumerate(zip(prob_est_all, cols_all_connected, n_matrices)):
            for j, (p2, c2, n2) in enumerate(zip(prob_est_all, cols_all_connected, n_matrices)):
                c12 = c1c2[i,j] #np.multiply(c1,c2)
                if sum(c12) > 0 and i != j:
                    if sum(np.multiply(p2, c12)) == 0 or sum(np.multiply(p1, c12)) == 0:
                        print(f"p1: {np.multiply(p1, c12)}")
                        print(p1)
                        print(c1)
                        print(f"p2: {np.multiply(p2, c12)}")
                        print(p2)
                        print(c2)

                    deltaG_matrix[i,j] = sum(np.multiply(p2, c12))/sum(np.multiply(p1, c12))
                    #pop_ratio = sum(np.multiply(p2, c12))/sum(np.multiply(p1, c12))
                    #deltaG_matrix[i,j] = -kT*np.log(pop_ratio)
                    #I have no theoretical justification for this specific weighting scheme
                    landscape_comparison_weights[i,j] = n1*n2*sum(c12) #weight by number of matrices used to calculate each estimate and number of overlapping states
                    #not including n1 and n2 made things worse anecdotally

    deltaG_matrix = np.nan_to_num(-kT*np.log(deltaG_matrix))

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

    cols_mat = np.stack(cols_all_connected)

    states_sampled = np.where(np.sum(cols_mat, axis=0) > 0)[0]
    states_sampled_1hot = np.where(np.sum(cols_mat, axis=0) > 0, 1, 0)

    #print(cols_mat.shape)

    n_matrices_tiled = np.tile(np.array(n_matrices), (n_states,1)).transpose()
    #print(n_matrices_tiled)
    #print(np.tile(np.array(n_matrices), (cols_mat.shape[1],1)).transpose().shape)
    weight_matrix = cols_mat #np.multiply(cols_mat, n_matrices_tiled)

    deltaG_average_sampled = np.average(np.nan_to_num(deltaG_shifted[:,states_sampled], posinf=0.0, neginf=0.0), axis=0, weights=weight_matrix[:,states_sampled])

    msm_v3_pops_sampled = np.exp(-deltaG_average_sampled/kT)
    z = np.sum(msm_v3_pops_sampled)
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
    
    plt.plot(states_sampled, deltaG_average_sampled+kT*np.log(z), linewidth=2, color='black', zorder = 999999)
    
    #plt.plot(deltaG_shifted[-1], linewidth=2, color='grey', linestyle="dashed", zorder = 99999)

    true_populations, true_energies = system.normalized_pops_energies(kT, bincenters)

    plt.plot(-kT*np.log(true_populations), linewidth=2, color='red', zorder = 999998)

    plt.show()

    #sys.exit(0)

    return msm_v3_pops_all





    ################################# trimmings #####################################

        # print("-------------------------------------------------------")
        # print(pops_msm_i_allstates)
        # print(pops_msm_j_allstates)

        # #print(pops_msm_i)

        # pops_msm_i_allstates = np.zeros(n_states)
        # k=0
        # for i in range(n_states):
        #     if cols_1_1hot[i] == 1:
        #         pops_msm_i_allstates[i] = pops_msm_i[k]
        #         k+=1

    # #all reweighted transition probability matrices, for averaging
    # Pu_all = [] 

    # #calculate reweighted TPMs for every round
    # for t in range(len(trjs)-1):

    #     #count transitions
    #     transition_counts = np.zeros((n_states, n_states))
    #     for tr in transitions[t]:
    #         transition_counts[tr[1]][tr[0]] += 1

    #     #normalize transition count matrix to transition rate matrix
    #     #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
    #     # so that the probability associated with each element of X(t) is preserved 
    #     # (though not all at one index) when X(t) is multiplied by the TPM
    #     tpm = normalize(transition_counts, axis=0, norm='l1')

    #     #project vectors from unbiased basis into the basis with the metadynamics bias V
    #     #[:, np.newaxis] makes a row vector into a column vector
    #     reweight_matrix = np.outer(grid_weights[t], np.reciprocal(grid_weights[t]))

    #     Pu = np.multiply(reweight_matrix, tpm)

    #     Pu_all.append(Pu)

    # puv_arr = np.array(Pu_all)

    # puv_mean = np.mean(puv_arr, axis=0)

    # #calculate first eigenvector corresponding to equilibrium probability

    # pops_msm_v2 = MSM_methods.tpm_2_eqprobs(puv_mean)
    # pops_msm_v2 /= np.sum(pops_msm_v2)


#seems to have a bug; written by chatGPT
# def minimize_weighted_G2(G0, W):
#     """
#     Solve for v that minimizes sum_{ij} W_ij (G0_ij + v_i - v_j)^2

#     Parameters
#     ----------
#     G0 : (n, n) ndarray
#         Base matrix G_0
#     W : (n, n) ndarray
#         Weight matrix (assumed non-negative)

#     Returns
#     -------
#     v : (n,) ndarray
#         Minimizing vector v, with gauge sum(v) = 0
#     """
#     G0 = np.asarray(G0, dtype=float)
#     W = np.asarray(W, dtype=float)

#     n = G0.shape[0]
#     assert G0.shape == (n, n)
#     assert W.shape == (n, n)

#     # Row and column sums of W
#     row_sum = W.sum(axis=1)
#     col_sum = W.sum(axis=0)

#     # Construct Laplacian-like matrix L
#     L = np.zeros((n, n))

#     # Diagonal
#     L[np.diag_indices(n)] = row_sum + col_sum

#     # Off-diagonals
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 L[i, j] = -(W[i, j] + W[j, i])

#     # Construct RHS b
#     b = (W.T * G0).sum(axis=1) - (W * G0).sum(axis=1)

#     # Impose gauge: sum(v) = 0
#     # Replace last row with constraint
#     L[-1, :] = 1.0
#     b[-1] = 0.0

#     # Solve
#     v = solve(L, b)

#     return v