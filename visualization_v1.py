import matplotlib.pyplot as plt

def plot_landscape_estimate(bincenters, est_populations, true_populations, title):
    for i, esp in enumerate(est_populations):
        plt.plot(bincenters, esp, color = str(0.8 - 0.6*i/len(est_populations)))

    plt.plot(bincenters, true_populations, color="black")
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel("population")
    plt.show()