import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
sys.path.append('.')
import tools.plot_setup


WT_LIST = '../resources/csv/wt_list.csv'


if __name__ == '__main__':
    
    # List of all wind turbines
    cols = ['power_plant', 'lat', 'lon', 'nominal_power']
    wt_list = pd.read_csv(WT_LIST, usecols=cols)

    # Aggregation by power plants, taking the mean for latitute, and longitude
    wt_farms = wt_list.groupby(['power_plant']).mean()

    power_classes = wt_farms['nominal_power'].astype(int).unique()
    classes_upper_bound = np.max(power_classes) + 1

    # Plot the map of all power plants of Wallonia, with colorbar for power
    fig, ax = plt.subplots()
    cs = ax.scatter(wt_farms['lon'], wt_farms['lat'],
                    c=wt_farms['nominal_power'],
                    cmap=cm.get_cmap('tab20', classes_upper_bound))

    cbar = fig.colorbar(cs, ax=ax, boundaries=range(classes_upper_bound))
    cbar.ax.set_ylabel('Wind Turbine Power [MW]')

    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    ax.set_title('Power plants in Wallonia')

    plt.tight_layout()
    plt.savefig('../products/pdf/wt_farms.pdf', transparent=True)
    plt.show()
