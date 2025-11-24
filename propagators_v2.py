#propagators.py
#Jonathan Borowsky
#2/21/25

#Method for generating trajectories given a system (potential), temperature, and time step

################################################################################################################

import numpy as np
import time

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
def propagate(system, kT, trj_coords, timestep, nsegs, save_period):
  
    nd = np.array(trj_coords.shape) #actually the number of walkers times the number of dimensions   
    D = system.diffusion_coefficient
    
    trj_out = np.zeros((nsegs, trj_coords.shape[0], trj_coords.shape[1]))
    for i in range(nsegs):
    
        for step in range(save_period):
            trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

        trj_out[i] = trj_coords
        #trj_out.append(trj_coords.copy())

    return trj_out



#nsteps must be an integer multiple of save_period
def propagate_mtd(system, kT, trj_coords, timestep, nsegs, save_period, grid):
  
    t1 = time.time()
    nd = np.array(trj_coords.shape)   #this includes both the number of trajectories and the number of dimensions
    D = system.diffusion_coefficient
    
    #grid_rates = np.full(trj_coords.shape[0], grid.rate) 
    #np.ones(trj_coords.shape[0])*grid_rates #[grid.rate for i in trj_coords]

    #trj_out = []
    #w_out = []

    trj_out = np.zeros((nsegs, trj_coords.shape[0], trj_coords.shape[1]))
    w_out = np.zeros((nsegs, trj_coords.shape[0]))

    fc = 0
    updates = 0

    for i in range(nsegs):
    
        for step in range(save_period):

            aaa = grid.compute_forces(trj_coords)
            if len(aaa) == 5:
                print("aaa is None")
            
            t3 = time.time()
            trj_coords += D/kT * (system.F(trj_coords) + grid.compute_forces(trj_coords)) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)
            t4 = time.time()
            fc += t4 - t3
        
        # if i == 0:
        #     print(np.divide(system.F(trj_coords),grid.compute_forces(trj_coords)))
            # print(system.F(trj_coords))
            # print(grid.compute_forces(trj_coords))

        #trj_out.append(trj_coords.copy())
        trj_out[i] = trj_coords
        w_out[i] = grid.weights(trj_coords, kT)
        #w_out.append(grid.weights(trj_coords, kT))
        
        t5 = time.time()
        grid_rates = np.ones(trj_coords.shape[0])
        grid.update2(trj_coords, grid_rates)
        #grid.compute_forces(trj_coords)
        t6 = time.time()
        updates += t6 - t5

    t2 = time.time()
    # print(f"propagator total={t2-t1}")
    # print(f"force calculation={fc}")
    # print(f"updates={updates}")

    #print(trj_out[1].shape)
    return trj_out, w_out, grid



#same as propagate() but outputs only the last frame; 
#  avoids an extra layer of for loops when running WE
def propagate_save1(system, kT, trj_coords, timestep, nsteps):
  
    nd = np.array(trj_coords.shape)   
    D = system.diffusion_coefficient
    
    for step in range(nsteps):
        trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

    return trj_coords, None



def propagate_mtd_save1(system, kT, trj_coords, timestep, nsteps, grid, weights):
  
    nd = np.array(trj_coords.shape)   #this includes both the number of trajectories and the number of dimensions
    D = system.diffusion_coefficient
    
    for step in range(nsteps):
        trj_coords += D/kT * (system.F(trj_coords) + grid.compute_forces(trj_coords)) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

    #calculate weights of final coordinates
    w_out = grid.weights(trj_coords, kT)

    #Update grid. Each walker deposits a gaussian of height proportional to its weight.
    grid.update2(trj_coords, weights)

    return trj_coords, w_out