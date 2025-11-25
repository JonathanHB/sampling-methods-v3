import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_landscape_estimate(bincenters, est_populations, true_populations, title, xrange, yrange):
    
    #plt.figure(dpi=300)
    
    for i, esp in enumerate(est_populations):
        #plt.plot(bincenters, esp, color = str(0.8 - 0.6*i/len(est_populations))) #gray scale
        fade_rg = 0.8 - 0.8*i/len(est_populations)
        plt.plot(bincenters, esp, color = (1,fade_rg,fade_rg), linewidth=1)

    plt.plot(bincenters, true_populations, color="black")
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel("free energy (kT)")

    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.show()


def plot_landscape_estimates(le, observables_we, bincenters, kT, true_populations, true_energies, molecular_time_limit, aggregate_simulation_limit, min_communication_interval):

    energies = -kT*np.log(le[0])
    #visualization_v1.plot_landscape_estimate(bincenters, le[0], true_populations, le[1], xrange = (-22,20), yrange = (0,0.4))
    plot_landscape_estimate(bincenters, energies, true_energies, le[1], xrange = (-22,20), yrange = (-5,25))


    #region explored over time
    #prints energy estimate as red for too high and blue for too low

    energies = -kT*np.log(le[0])-true_energies

    masked_array = np.ma.array(energies, mask=np.isnan(energies))
    cmap = matplotlib.cm.bwr
    cmap.set_bad('grey',1.)

    plt.matshow(masked_array, interpolation='nearest', cmap=cmap, aspect=1, vmin=-8, vmax=8) #extent=[0,202,0,50]
    plt.colorbar()
    plt.title(le[1])
    plt.xlabel("bin")
    plt.ylabel(f"molecular time ({int(molecular_time_limit/min_communication_interval)} steps)")

    #TODO plot aggregate time at n_aggregate_marks intervals for non-WE simulations

    #plot aggregate time for WE simulations
    #TODO plot these as a second graph y axis
    if le[1][0:2] == "WE":
        n_aggregate_marks = 10
        for t_thresh in range(n_aggregate_marks):
            for ti, t in enumerate(observables_we[2]):
                if t > aggregate_simulation_limit * t_thresh/n_aggregate_marks:
                    plt.plot([0,200], [ti,ti], linestyle="dashed", color="lime")
                    labelnumber = "{:.1f}".format(np.round(t/1000000, 1))
                    plt.annotate(f"{labelnumber}M steps", (150, ti))
                    break

    #plt.imshow(energies)
    #plt.axis("equal")
    plt.show()


def plot_convergence(landscape_estimates_plot, kT, true_populations, true_energies, molecular_time_limit, n_timepoints):
    for le in landscape_estimates_plot:

        RMS_energy_errors = []

        for lei in le[0]:
            energies = -kT*np.log(lei)
            RMS_energy_errors.append(np.sqrt(np.mean([(e-te)**2 for e, te, p in zip(energies, true_energies, lei) if p > 0])))

        #mean_absolute_error = [np.mean([abs(epi-pi) for pi, epi in zip(true_populations, est_pops)]) for est_pops in le[0]]
        times = [int(round((t+1)*(molecular_time_limit/n_timepoints))) for t in range(len(RMS_energy_errors))]
        
        plt.plot(times, RMS_energy_errors)


    plt.legend([le[1] for le in landscape_estimates_plot])
    plt.xlabel("time")
    plt.ylabel("root mean squared error (kT)")

    plt.xlim(0,molecular_time_limit)
    plt.ylim(0,10)
    #plt.ylim(0,0.1)

    plt.show()
