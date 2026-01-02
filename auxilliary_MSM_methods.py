import numpy as np

#TODO: the contents of this class should be incorporated into either MSM_methods.py or mtd_estimators.py and this class should be deprecated

def tpm_2_eqprobs(msm_tpm, eq_eigenvalue, print_convergence_time=False):

    #get tpm eigenvalues and eigenvectors to find state probabilities
    msm_eigs = np.linalg.eig(msm_tpm)
        
    #make sure we're getting the correct eigenvector with eigenvalue 1
    nfigs = 12
    
    eig1_ind = -1
    
    for ex, eigenvalue in enumerate(msm_eigs[0]):
        if np.round(eigenvalue, nfigs) == eq_eigenvalue:
            if eig1_ind != -1:
                #note that this has since been demonstrated not to occur for connected systems,
                # but serves as a warning for disconnected ones, which do have multiple 1 eigenvalues
                print("warning: multiple eigenvalues equal to 1 detected, one was selected arbitrarily")
                print(f"eigenvalues were {msm_eigs[0]}")
            eig1_ind = ex

    if eig1_ind == -1:
        print(f"error: no eigenvalue is {eq_eigenvalue} to within {nfigs} significant figures")
        
        eig1 = min(msm_eigs[0], key=lambda x:abs(x-eq_eigenvalue))
        eig1_ind = np.where(msm_eigs[0] == eig1)
        
        print(f"using eigenvalue {eig1}")

    eig0_raw = msm_eigs[1][:,eig1_ind] #this is the eigenvector associated with the eigenvalue 1
    
    #normalize so that total population = 1
    eig0 = eig0_raw/sum(eig0_raw)
    #change eigenvector to a column vector for right multiplication by tpm
    eig0c = eig0.reshape((len(eig0), 1))

    #repeatedly multiply the unrefined eigenvector by the transition probability matrix until it stops changing
    #in order to get rid of numerical errors from np.linalg.eig
    #however at times this looks like it may be chasing floating point errors
    converged = False
    refinement_rounds = 99
    maxerror = -1
            
    for r in range(refinement_rounds):
        #time evolve unrefined eigenvector
        eig0c_buffer = np.dot(msm_tpm,eig0c)

        #calculate change in state vector
        fractional_errors = (eig0c-eig0c_buffer)/eig0c
        
        maxerror = max(abs(max(fractional_errors)[0]), abs(min(fractional_errors)[0]))
        if maxerror < 10**-nfigs and min(eig0c)[0] >= 0:
            if print_convergence_time:
                print(f"eigenvector converged to within 10^{-nfigs} after {r} rounds")
            converged = True
            break
            
        eig0c = eig0c_buffer
            
    if not converged:
        print(f"error: eigenvector failed to converge after {refinement_rounds} rounds; \
maximum fractional error of any component = {maxerror}")
    
    #some numerical inputs yield complex eigenvalues and eigenvectors but the equilibrium probability vector
    # should be real; verify that it is
    if not all(np.imag(eig0c).flatten() == np.zeros(len(eig0c))):
        print("error: nonzero complex components detected")
        
    return np.real(eig0c)



def tpm_2_eqprobs_v3(msm_tpm, print_convergence_time=False):

    #get tpm eigenvalues and eigenvectors to find state probabilities
    msm_eigs = np.linalg.eig(msm_tpm)
        
    #make sure we're getting the correct eigenvector with eigenvalue 1
    nfigs = 12
    
    eig1_ind = -1
    eig1 = 1
    
    for ex, eigenvalue in enumerate(msm_eigs[0]):
        if np.round(eigenvalue, nfigs) == 1:
            if eig1_ind != -1:
                #note that this has since been demonstrated not to occur for connected systems,
                # but serves as a warning for disconnected ones, which do have multiple 1 eigenvalues
                print("warning: multiple eigenvalues equal to 1 detected, one was selected arbitrarily")
                print(f"eigenvalues were {msm_eigs[0]}")
            eig1_ind = ex

    if eig1_ind == -1:
        #print(f"warning: no eigenvalue is 1 to within {nfigs} significant figures")
        
        eig1 = max(msm_eigs[0]) #min(msm_eigs[0], key=lambda x:abs(x-1))
        eig1_ind = np.where(msm_eigs[0] == eig1)
        
        # if eig1 < 0:
        #     print(f"eigenvalue = {eig1}")
        #     print(f"eigenvector = {msm_eigs[1][:,eig1_ind]}")

        #print(f"using eigenvalue {eig1}")

    eig0_raw = msm_eigs[1][:,eig1_ind] #this is the eigenvector associated with the eigenvalue 1
    
    #normalize so that total population = 1
    eig0 = eig0_raw/sum(eig0_raw)
    #change eigenvector to a column vector for right multiplication by tpm
    eig0c = eig0.reshape((len(eig0), 1))

    #repeatedly multiply the unrefined eigenvector by the transition probability matrix until it stops changing
    #in order to get rid of numerical errors from np.linalg.eig
    #however at times this looks like it may be chasing floating point errors
#     converged = False
#     refinement_rounds = 99
#     maxerror = -1
            
#     for r in range(refinement_rounds):
#         #time evolve unrefined eigenvector
#         eig0c_buffer = np.dot(msm_tpm,eig0c)/eig1

#         #calculate change in state vector
#         fractional_errors = (eig0c-eig0c_buffer)/eig0c
        
#         maxerror = max(abs(max(fractional_errors)[0]), abs(min(fractional_errors)[0]))
#         if maxerror < 10**-nfigs and min(eig0c)[0] >= 0:
#             if print_convergence_time:
#                 print(f"eigenvector converged to within 10^{-nfigs} after {r} rounds")
#             converged = True
#             break
            
#         eig0c = eig0c_buffer
            
#     if not converged:
#         print(f"error: eigenvector failed to converge after {refinement_rounds} rounds; \
# maximum fractional error of any component = {maxerror}")
#         print(f"eigenvalue {eig1}")
#         print(f"eigenvalue {msm_eigs[0]}")
    
    #some numerical inputs yield complex eigenvalues and eigenvectors but the equilibrium probability vector
    # should be real; verify that it is
    if not all(np.imag(eig0c).flatten() == np.zeros(len(eig0c))):
        print("error: nonzero complex components detected")
        
    return np.real(eig0c)