import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
import tools.plot_setup
import cache.wind_ls


if __name__ == '__main__':
    
    # Retrieve the whole learning set
    X, power, time = cache.wind_ls.get_learning_set()

    # Compute the mean wind speed along each power plant
    wind_speed = X[:, 0::2]
    mean_wind_speed = np.mean(wind_speed, axis=1)

    # Plot the density of observations in the 'mean wind speed' - 'power' space
    fig, ax = plt.subplots()
    cs = ax.hexbin(mean_wind_speed, power, bins='log', gridsize=(50, 30))
    
    cbar = fig.colorbar(cs, ax=ax)
    cbar.ax.set_ylabel('Density')

    ax.set_xlabel('Wind')
    ax.set_ylabel('Power')
    plt.tight_layout()
    plt.savefig('../products/pdf/wallonia_power_curve.pdf')
    plt.show()
