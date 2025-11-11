#energy_landscapes.py
#Jonathan Borowsky
#2/21/25

#classes representing different systems to be simulated
#these provide energy landscapes, forces, diffusion coefficients,
# kinetically appropriate macrostate definitions,
# and convenient fixed equal width bins and starting coordinates

################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


#superclass of 1d potential functions
#this contains all the functions which are useful for a system object to have 
# and for which the structure of the algorithm does not depend on the potential or other details of the system
class potential_well_1d():

    def __init__(self, potential, macro_class, standard_analysis_range):
        self.potentiall = potential
        self.macro_classs = macro_class
        self.standard_analysis_rangee = standard_analysis_range #is this actually necessary??

    #determine which ensemble a trajectory currently ensemble e should be in upon moving to coordinate x
    def ensemble_class(self, x, e):  
        ms = self.macro_classs(x)
        if ms != -1:
            return ms
        else:
            return e

    #calculate equilibrium populations and energies for a given set of bins
    # by assuming energy is roughly constant across each bin
    # This is a good approximation for most systems; 
    # if your bins are too large for it to hold it's usually a good idea to make them smaller 
    # instead of integrating across them with the method below
    def normalized_pops_energies(self, kT, bincenters):
        #assume equal bin widths
        binwidth = bincenters[1]-bincenters[0]

        pops_nonnorm = [np.exp(-self.potentiall(x)/kT) for x in bincenters]
        z = sum(pops_nonnorm)
        pops_norm = [p/z for p in pops_nonnorm]
    
        energies_norm = [-kT*np.log(p/(z*binwidth)) for p in pops_nonnorm]

        return pops_norm, energies_norm

    #compute equilibrium populations of the given bins by integrating across them
    # this should be more accurate than just using the center point of the bin
    # the increase in accuracy provided by this method seems to be entirely unnecessary in practice
    #tolerance is the permitted energy difference between the edges of a sub-bin in kT
    #bin_boundaries are assumed to increase monotonically
    def compute_true_populations(self, bin_boundaries, kT, tolerance = 0.01):

        bin_centers = []
        bin_populations = []

        for i in range(len(bin_boundaries)-1):
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1])/2)

            #figure out how many sub-bins the bin must be divided into for the potential across each bin to be roughtly constant
            #This method assumes negligible curvature and will fail for bins with equal edge energies which curve up or down in between
            #a more general approach would be to randomly sample points in each bin and then average or sum somehow
            energy_gap = abs(self.potentiall(bin_boundaries[i+1])-self.potentiall(bin_boundaries[i]))/kT
            n_subbins = max(int(np.ceil(energy_gap/tolerance)), 1)
            
            bin_pop = 0
            subbin_width = (bin_boundaries[i+1] - bin_boundaries[i])/n_subbins
            
            for sbx in np.linspace(bin_boundaries[i]+subbin_width/2, bin_boundaries[i+1]-subbin_width/2, n_subbins):
                bin_pop += np.exp(-self.potentiall(sbx)/kT)*subbin_width
                
            bin_populations.append(bin_pop)

        z = sum(bin_populations)
        bin_populations = [bp/z for bp in bin_populations]
        
        return bin_centers, bin_populations
    
    #for visualization to check that you've written the potential right
    def plot_quantity(self, quantity): 
        x = np.linspace(self.standard_analysis_rangee[0], self.standard_analysis_rangee[1], 100)
        plt.plot(x, [quantity(i) for i in x])

    #return bins for analysis of each energy landscape, 
    # including end bins for anything outside the standard bin range
    def analysis_bins_1d(self, nbins):
        
        step = (self.standard_analysis_rangee[1][0]-self.standard_analysis_rangee[0][0])/nbins
    
        binbounds = np.linspace(self.standard_analysis_rangee[0][0], self.standard_analysis_rangee[1][0], nbins+1)
        bincenters = np.linspace(self.standard_analysis_rangee[0][0]-step/2, self.standard_analysis_rangee[1][0]+step/2, nbins+2)

        return binbounds, bincenters, step
    
    def bin_trj_nd(self, nbins, trj):
        
        steps = [(self.standard_analysis_rangee[1][i]-self.standard_analysis_rangee[0][i])/nbins for i in range(len(self.standard_analysis_rangee[1]))]


#a double well constructed using a quartic and quadratic potential
class unit_double_well(potential_well_1d):
    #MFPT(10 frame save frequency) = ~800 steps

    def potential(self, x):
        return x**4 - x**2
        
    def F(self, x):
        return -4*x**3 + 2*x
    
    def macro_class(self, x):
        thr = 0.7 #1/np.sqrt(2)
        if x[0] <= -thr:
            return 0
        elif x[0] >= thr:
            return 1
        else:
            return -1

            
    def __init__(self):
        self.diffusion_coefficient = 1
        self.n_macrostates = 2
        self.standard_init_coord = [-1/np.sqrt(2)]
        self.standard_analysis_range = [[-2],[2]]
        self.start_from_index = False
        super().__init__(self.potential, self.macro_class, self.standard_analysis_range)


#a system of several wells of similar energies constructed using a sinusoidal and a quartic potential
class unit_sine_well(potential_well_1d):
    #MFPT(10 frame save frequency) = ~70000 steps

    def potential(self, x):
        return 0.0001*x**4 + np.cos(x)
        
    def F(self, x):
        return 0.0001*-4*x**3 + np.sin(x)
    
    def macro_class(self, x):
        thr = 2*np.pi
        if x[0] < -thr:
            return 0
        elif x[0] > thr:
            return 1
        else:
            return -1
        
    def __init__(self):
        self.diffusion_coefficient = 1
        self.n_macrostates = 2
        self.standard_init_coord = [-3*np.pi]
        self.standard_analysis_range = [[-20],[20]]
        self.start_from_index = False
        super().__init__(self.potential, self.macro_class, self.standard_analysis_range)


#this is a stub whose functionality is mostly implemented in energy_landscapes_gaussian.py
# #build an energy landscape with a specified set of minima, and transition states.
#         #this landscape is to be represented by a set of points placed randomly in n-dimensional space at the specified state_density
#         #temperature is used to construct the transition probability matrix (it should cancel out elsewhere? <--TODO verify this)
#         #noise spectrum is the amplitude of noise to apply as a function of the spatial frequency of the noise (i.e. a low frequency noise is applied with wide gaussians)
# def build_landscape(n_dim, minima_coords, minima_energies, ts_energies, state_density, temperature, noise_spectrum):

#     box_min = np.min(minima_coords, axis=0)
#     box_max = np.max(minima_coords, axis=1)
#     print(box_min)
#     print(box_max)
