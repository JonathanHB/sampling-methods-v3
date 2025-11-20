import numpy as np
import itertools
import sys
import time

import binning_v1
import matplotlib.pyplot as plt

#idea: what if we ran metadynamics with evaporating/eroding gaussians? Like if every new gaussian had the same weight but old ones gradually shrunk. This might not work so well with the grid though.

#METHOD OUTLINE
#objects:
# 1. grid: a grid with a potential in each grid cell
#      constructor: takes the output of the voxel bin construction function in analysis
#      compute_forces: returns a list of forces on atoms in each grid cell
#      add_point(x, gaussian_width, gaussian_height): adds a gaussian potential to the grid centered at x. This could also take a weight to go with gaussian_height
      
#functions:
# a propagator that takes the grid boundaries and forces from grid/compute_forces and the bin assigner from analysis and updates the positions of the atoms
# it's like a regular propagator but adds f(bin(x)) to the force

#method: run westpa with a single grid object. The object's forces are passed to the propagator at the start of each round 
# and each walker's final coordinate is added to the grid at the end of the round


class grid():

    def __init__(self, bounds, nbins_total, rate=1, dT=99999, stdevs=[]):
        
        self.bounds = bounds
        #print(self.bounds)
        self.dim = len(bounds[0])
        self.rate = rate
        self.dT = dT

        if stdevs == []:
            self.stdevs = [0.025*(xmax-xmin) for xmax, xmin in zip(bounds[1], bounds[0])]
        else:
            self.stdevs = stdevs
        self.invstds = np.reciprocal(self.stdevs)

        self.bincenters, self.binwidth, self.nbins, self.nbins_total_scalar_actual, self.binbounds, self.ndim, self.prods_higher = binning_v1.construct_voxel_bins(bounds, nbins_total)
        #print(self.nbins)
        #print(f"bincenters {self.bincenters}")
        self.grid = np.zeros(self.nbins)
        self.forcegrids = -np.gradient(self.grid, self.binwidth).reshape([self.dim]+self.nbins)

        #construct_voxel_bins should probably return this instead since it certainly doesn't need to be calculated every time this function is called
        self.grid_coords = [gci for gci in itertools.product(*[[r for r in range(nbi)] for nbi in self.nbins])]


    # super slow do not use
    # Update the grid based on the new position
    def update(self, trjs, weights):
        t1 = time.time()
        #calculate the magnitude of the gaussian associated with every trajectory point in every grid bin
        for bc, gc in zip(self.bincenters, self.grid_coords):
            for x, w in zip(trjs, weights):

                expterm = np.multiply(np.array(bc)-np.array(x), self.invstds)
                self.grid[gc] += w * np.prod(self.invstds) * (2*np.pi)**(-self.dim/2) * np.exp(-0.5*np.dot(expterm, expterm))
        t2 = time.time()
        print(f"potential update={t2-t1}")
        #updating forces here is only practical because all dimensions are part of the progress coordinate; 
        # if there were many other dimensions orthogonal to the progress coordinate, I don't think it would work
        self.forcegrids = -np.gradient(self.grid, self.binwidth).reshape([self.dim]+self.nbins)
        t3 = time.time()
        print(f"force update={t3-t2}")


    def compute_forces(self, trjs):

        bin_inds, nd_inds = binning_v1.bin_to_voxels_timeslice(self.ndim, self.binbounds, self.prods_higher, trjs)

        #---------extract the force at the grid cell occupied by the particle in each trajectory----------
        #For an n-dimensional metadynamics progress coordinate, we have:
        #   self.forcegrids: an (n+1)d array of force vector components at each grid cell
        #                    the 0th axis of which indexes the force components in each dimension
        #                    (this can also be thought of an nd array of of force vectors at each grid cell, 
        #                    although that analogy does not capture the order of the axes correctly)
        #   nd_inds: a 2d array of the grid coordinates of each particle (of shape [number_of_particles, n])
        #We want to extract the 1d columns of forcegrids (the force vectors, extending along the 0th axis of the array) 
        #   at the coordinates on the n-dimensional face spanned by axes 1-(n+1) specified by nd_inds 
        #   as a list of 1d vectors (which are the forces on each particle).
        #To do this we interate over the 0th axis of forcegrids
        #   This allows us to extract scalars from a n n-dimensional array given n-dimensional coordinates
        #   rather than trying to extract 1-dimensional vectors from a (n+1)-dimensional array given n-dimensional coordinates.
        #   TODO: could we stack nd_inds to get n+1 dimensional coordinates and extract the force vectors directly?
        #   At each iteration we extract the i-th component of the force on each particle.
        #   The resulting lists are then stacked and transposed to get the whole force vector for each particle.
        #see https://stackoverflow.com/questions/69865664/get-values-from-an-array-given-a-list-of-coordinates-locations-with-numpy

        forces = np.array([fgc.take(np.ravel_multi_index(nd_inds.T, fgc.shape)) for fgc in self.forcegrids]).transpose()

        return forces
    

    #get the metadynamics energy and corresponding weight at each grid point
    #BEWARE that the output is not normalized
    def weights(self, trjs, kT):

        bin_inds, nd_inds = binning_v1.bin_to_voxels_timeslice(self.ndim, self.binbounds, self.prods_higher, trjs)
        energies = self.grid.take(np.ravel_multi_index(nd_inds.T, self.grid.shape))

        return np.exp(energies/kT)
    

    #TODO: clean up and comment this section properly
    #fully vectorized parallel grid update code for multiple walkers and PC dimensions
    def update2(self, coords, weights):

        t1 = time.time()
        mgrid_args = tuple(slice(self.bincenters[0][d], self.bincenters[-1][d], self.nbins[d]*1j) for d in range(self.dim))
        mg = np.mgrid[mgrid_args]

        #bincenter matrix
        mgrid_tile_dims = [len(coords)]+[1 for i in range(self.ndim+1)]
        mgrid_tiled = np.tile(mg, mgrid_tile_dims) #(len(coords),1,1,1))
        #print(mgrid_tiled.shape)
        # print(mgrid_tiled)

        # print(mg)
        # sys.exit(0)
        #coords = np.array([[-1,0],[3,4],[0,-4]]) #n_walkers x n_dimensions

        t2 = time.time()
        #print(f"mgrid {t2-t1}")

        transpose_arg_s0 = [0,self.dim+1]+[i+1 for i in range(self.dim)]
        #print(transpose_arg_s0)

        #sigma matrix
        s0 = np.tile(self.stdevs, (len(coords),*tuple(self.nbins),1)).transpose(*transpose_arg_s0) #(0,3,1,2)
        #print(s0.shape)
        #sys.exit(0)
        # print(s0)

        t3 = time.time()
        #print(f"sigma {t3-t2}")

        #weight matrix, currently used for well-tempered metadynamics
        particle_weights = self.weights(coords, -self.dT)

        transpose_arg_w0 = [self.dim]+[i for i in range(self.dim)]

        w0 = np.tile(particle_weights, (*tuple(self.nbins),1)).transpose(transpose_arg_w0)
        # print(w0.shape)


        transpose_arg_x0 = [self.dim, self.dim+1]+[i for i in range(self.dim)]
        #print(transpose_arg_x0)

        #walker coordinate matrix
        x0 = np.tile(coords, (*self.nbins,1,1)).transpose(*transpose_arg_x0) #(2,3,0,1) #= np.full((10,10), 5)
        #print(x0.shape)
        # print(x0)

        t4 = time.time()
        #print(f"coords {t4-t3}")

        #the exponent is a matrix M which in numpy yields a matrix of exponents of the elements of M (i.e. elementwise exponentiation)
        expcomps = np.exp(-((x0-mgrid_tiled)/(2*s0))**2)
        # print(expcomps)
        #print(expcomps.shape)
        t5 = time.time()
        #print(f"exp {t5-t4}")

        exps = np.product(expcomps, axis = 1)
        # print(exps)
        # print("exps")
        # print(exps.shape)

        tempered_exps = np.multiply(exps, w0)

        # print(tempered_exps.shape)

        #this is a great way of visualizing gaussian deposition for 1d PCs
        show_tempering = False
        if show_tempering:
            plt.imshow(exps)
            plt.show()
            plt.imshow(tempered_exps)
            plt.show()
            print("-----------------------------------------------------------------")
        # for e in exps:
        #     plt.imshow(e, vmax=1.1, extent = np.array(self.bounds).transpose().flatten(), aspect = (bounds[1][0]-bounds[0][0])/(bounds[1][1]-bounds[0][1]))
        #     #plt.axis("equal")
        #     plt.show()
        # sys.exit(0)

        exptot = np.sum(tempered_exps, axis = 0)
        # print(exptot)
        # print(exptot.shape)
        # plt.plot(exptot)
        # plt.show()

        t6 = time.time()
        #print(f"sumprod {t6-t5}")
        
        # print(exptot.shape)
        # print(self.rate.shape)

        self.grid += exptot*self.rate

        self.forcegrids = -np.gradient(self.grid, self.binwidth).reshape([self.dim]+self.nbins)


        # print(np.max(exptot))
        # plt.imshow(exptot, vmax=1.1, extent = np.array(self.bounds).transpose().flatten(), aspect = (bounds[1][0]-bounds[0][0])/(bounds[1][1]-bounds[0][1]))
        # plt.show()
