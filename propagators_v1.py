#propagators.py
#Jonathan Borowsky
#2/21/25

#Method for generating trajectories given a system (potential), temperature, and time step

################################################################################################################

import numpy as np

#parameters
#   trj_coords: list of floats: initial coordinates of the trajectories on the progress coordinate
#   F: function of a float returning a float: 
#      the negative derivative of free energy function with respect to the progress coordinate as a function of the progress coordinate
#   D: float: brownian diffusion coefficient
#   kT: float: Boltzmann's constant times the temperature
#   timestep: float: the size of the timestep used for propagation
#   nsteps: nonnegative int: how many time steps to propagate for

#returns
#   trj_out: list of arrays: the coordinates of the trajectories at each time step
#      trj_out has size [nsteps//save_period, trj_coords.shape[0], trj_coords.shape[1]]

#Brownian diffusion
#nsteps must be an integer multiple of save_period
def propagate(system, kT, trj_coords, timestep, nsteps, save_period):
  
    nd = np.array(trj_coords.shape) #actually the number of walkers times the number of dimensions   
    D = system.diffusion_coefficient
    
    trj_out = np.zeros((nsteps//save_period, trj_coords.shape[0], trj_coords.shape[1]))
    for i in range(nsteps//save_period):
    
        for step in range(save_period):
            trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

        trj_out[i] = trj_coords
        #trj_out.append(trj_coords.copy())

    return trj_out



#nsteps must be an integer multiple of save_period
def propagate_mtd(system, kT, trj_coords, timestep, nsteps, save_period, grid):
  
    nd = np.array(trj_coords.shape)   #this includes both the number of trajectories and the number of dimensions
    D = system.diffusion_coefficient
    
    grid_rates = [grid.rate for i in trj_coords]

    trj_out = []
    w_out = []
    
    for i in range(nsteps//save_period):
    
        for step in range(save_period):
            trj_coords += D/kT * (system.F(trj_coords) + grid.compute_forces(trj_coords)) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)
        
        trj_out.append(trj_coords.copy())
        w_out.append(grid.weights(trj_coords, kT))
        
        grid.update(trj_coords, grid_rates)
        #grid.compute_forces(trj_coords)

    #print(trj_out[1].shape)
    return trj_out, w_out, grid



#same as propagate() but outputs only the last frame; 
#  avoids an extra layer of for loops when running WE
def propagate_save1(system, kT, trj_coords, timestep, nsteps):
  
    nd = np.array(trj_coords.shape)   
    D = system.diffusion_coefficient
    
    for step in range(nsteps):
        trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

    return trj_coords