import matplotlib.pyplot as plt

def plot_landscape_estimate(bincenters, est_populations, true_populations, title, xrange, yrange):
    
    plt.figure(dpi=300)
    
    for i, esp in enumerate(est_populations):
        #plt.plot(bincenters, esp, color = str(0.8 - 0.6*i/len(est_populations))) #gray scale
        fade_rg = 0.8 - 0.8*i/len(est_populations)
        plt.plot(bincenters, esp, color = (1,fade_rg,fade_rg), linewidth=0.5)

    plt.plot(bincenters, true_populations, color="black")
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel("population")

    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.show()