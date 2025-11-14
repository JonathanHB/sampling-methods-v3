import numpy as np
import itertools

#----------------------------------------------------------------------------------------------------------------

def construct_voxel_bins(analysis_range, nbins):

    ndim = len(analysis_range[0])

    boxlengths = [xmax-xmin for xmax, xmin in zip(analysis_range[1], analysis_range[0])]
    boxcenters = [(xmax+xmin)/2 for xmax, xmin in zip(analysis_range[1], analysis_range[0])]

    binwidths = []
    for bl in boxlengths:
        binwidths.append(bl*(nbins*np.product([bl/blj for blj in boxlengths]))**(-1/ndim))

    #make bins the same size in each dimension, 
    # preventing anisotropies from arising from the fact that the analysis box edge lengths may not be in an integer ratio
    binwidth = np.mean(binwidths) 

    #calculate bin centers and boundaries
    binbounds = []
    bincenters = []
    nbins = []

    for d in range(ndim):
        nbins_d = int(np.ceil(boxlengths[d]/binwidth))
        nbins.append(nbins_d+2)

        rmin = boxcenters[d]-binwidth*nbins_d/2
        rmax = boxcenters[d]+binwidth*nbins_d/2

        binbounds.append(np.linspace(rmin, rmax, nbins_d+1))
        bincenters.append(np.linspace(rmin-binwidth/2, rmax+binwidth/2, nbins_d+2))

    bincenters_flat = list(itertools.product(*bincenters))

    actual_nbins = np.product(nbins)
    prods_higher = [np.product(nbins[i:]) for i in range(1,len(nbins))] + [1]

    return bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher



def construct_voxel_bins_2_widths(analysis_range, nbins):

    ndim = len(analysis_range[0])

    boxlengths = [xmax-xmin for xmax, xmin in zip(analysis_range[1], analysis_range[0])]
    boxcenters = [(xmax+xmin)/2 for xmax, xmin in zip(analysis_range[1], analysis_range[0])]

    binwidths = []
    for bl in boxlengths:
        binwidths.append(bl*(nbins*np.product([bl/blj for blj in boxlengths]))**(-1/ndim))

    #make bins the same size in each dimension, 
    # preventing anisotropies from arising from the fact that the analysis box edge lengths may not be in an integer ratio
    binwidth = np.mean(binwidths) 

    #calculate bin centers and boundaries
    binbounds = []
    bincenters = []
    nbins = []

    for d in range(ndim):
        nbins_d1 = int(np.ceil(boxlengths[d]/binwidth)*2/3)
        nbins_d2 = int(np.ceil(boxlengths[d]/binwidth)*1/3)

        nbins.append(nbins_d1+nbins_d2+1)

        rmin = boxcenters[d]-binwidth*nbins_d1*3/2
        rmax = boxcenters[d]+binwidth*nbins_d2*3/4

        binbounds.append(np.concatenate((np.linspace(rmin, 0, nbins_d1)[:-1], np.linspace(0, rmax, nbins_d2+1))))
        bincenters.append(np.concatenate((np.linspace(rmin-binwidth/2, -binwidth*2/3, nbins_d1+1)[:-1], np.linspace(binwidth/3, rmax+binwidth/2, nbins_d2+1))))

    bincenters_flat = list(itertools.product(*bincenters))

    actual_nbins = np.product(nbins)
    prods_higher = [np.product(nbins[i:]) for i in range(1,len(nbins))] + [1]

    binwidths = [binbounds[0][i+1]-binbounds[0][i] for i in range(len(binbounds[0])-1)]
    # plt.plot(bincenters[0][1:-1], binwidths)
    # plt.show()

    return bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher


#----------------------------------------------------------------------------------------------------------------

def bin_to_voxels(ndim, binbounds, prods_higher, trjs):

    #bin trajectories in each dimension
    binned_all = []

    for trj in trjs:

        binned_by_dim = []    
        for d in range(ndim):
            binned_by_dim.append(np.digitize([f[d] for f in trj], bins = binbounds[d]))
        
        binned_all.append(np.array(binned_by_dim))

    #prods_higher = [np.product(nbins[i:]) for i in range(1,len(nbins))] + [1]
    
    trjs_binned = [np.matmul(prods_higher, binned_by_dim) for binned_by_dim in binned_all]

    return trjs_binned


def bin_to_voxels_timeslice(ndim, binbounds, prods_higher, trjs):

    #bin trajectories in each dimension
    #an array of shape [n_dims x n_trjs]
    binned_trj_dim = np.array([np.digitize(trjdim, bins = binboundsdim) for trjdim, binboundsdim in zip(trjs.transpose(), binbounds)])

    #bin trajectories into 1D bins
    #an array of shape [n_trjs]
    trjs_binned_flat = np.matmul(prods_higher, binned_trj_dim).transpose()

    return trjs_binned_flat, binned_trj_dim.transpose()
