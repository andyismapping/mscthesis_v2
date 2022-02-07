# import modules
import glob
import numpy as np
import pandas as pd
from astropy.time import Time
import cdflib
import datetime
from astropy.table import Table
import math
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import sys
import gc

# defined functions
def earth_radius(B):
    B = math.radians(B)  # converting into radians
    a = 6378.137  # Radius at sea level at equator
    b = 6356.752  # Radius at poles
    c = (a ** 2 * math.cos(B)) ** 2
    d = (b ** 2 * math.sin(B)) ** 2
    e = (a * math.cos(B)) ** 2
    f = (b * math.sin(B)) ** 2
    R = math.sqrt((c + d) / (e + f))
    return R


def objective(x, a, b):
    return a * np.exp(b * x)


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# defined parameters
spatial_window = 5 # km
temporal_window = 15 # min
altitude_window = 100 # km

# imput files

# radar_list = glob.glob('../data/external/Madrigal/2018-2021/*/*.hdf5')
radar_list = glob.glob('/Users/andyara/Documents/GFZ/code/OnEs/data/external/Madrigal/2018-2021/MillstoneHillISRadar/*.hdf5')

radar_list.sort()

gf_ts = pd.read_csv('/Users/andyara/Documents/GFZ/code/OnEs/data/processed/GF_ts_complete.csv')
gf_ts['dates'] = (Time(gf_ts['CDF Epoch'].values, format='cdf_epoch')).datetime

def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


for radar_file in radar_list:

    # print(radar_file.split('/')[-1])

    try:

        radar = (Table.read(radar_file, path='Data/Table Layout')).to_pandas()
        radar['dates'] = Time(radar[['ut1_unix', 'ut2_unix']].mean(axis=1), format='unix').datetime

        radar_metadata = (Table.read(radar_file, path='Metadata/Experiment Parameters')).to_pandas()

        names = np.array([x.decode() for x in radar_metadata['name']])
        values = np.array([x.decode() for x in radar_metadata['value']])

        radar_metadata = pd.DataFrame({'name': names,
                                       'value': values})
        radar_lat = float(radar_metadata[radar_metadata['name'] == 'instrument latitude']['value'].values[0])
        radar_lon = float(radar_metadata[radar_metadata['name'] == 'instrument longitude']['value'].values[0])
        radar_lon = (((radar_lon + 180) % 360) - 180)

        gf = gf_ts

        # filter space
        gf = gf[((gf['Latitude'] <= radar_lat + spatial_window) & (gf['Latitude'] >= radar_lat - spatial_window))]
        gf = gf[((gf['Longitude'] <= radar_lon + spatial_window) & (gf['Longitude'] >= radar_lon - spatial_window))]

        if len(gf) > 0:
            # filter time
            gf = gf[((gf['dates'] <= radar['dates'].max() + datetime.timedelta(hours=6)) & (
                        gf['dates'] >= radar['dates'].min() - datetime.timedelta(hours=6)))]

            radar = radar[((radar['dates'] <= gf['dates'].max() + datetime.timedelta(minutes=temporal_window)) & (
                        radar['dates'] >= gf['dates'].min() - datetime.timedelta(minutes=temporal_window)))]

            # save original
            radar_old = radar


            # filter altitude
            Re = earth_radius(radar_lat)
            gf_alt = np.mean(gf['Radius'] * 0.001 - Re)

            radar = radar[((radar['gdalt'] <= gf_alt + altitude_window) & (radar['gdalt'] >= gf_alt - altitude_window))]

            if len(radar) > 0:

                # print(radar_file.split('/')[-1])
                #                 print('GF', gf.dates.min(), gf.dates.max())
                #                 print('RADAR', radar.dates.min(), radar.dates.max())
                #                 print(radar_mean)
                #                 print('--------------')

                # black profile
                radar_mean = pd.DataFrame({'gdalt': radar['gdalt'],
                                           'ne': radar['ne']})

                radar_mean = radar_mean.groupby(['gdalt']).mean()
                radar_mean_notsmooth = radar_mean

                Xresampled = np.arange(int(radar_mean.index.min()), int(radar_mean.index.max()), 10)

                radar_mean = radar_mean.reindex(radar_mean.index.union(Xresampled)).interpolate('spline', order=1).loc[
                    Xresampled]

                fig, ax = plt.subplots(figsize=(10, 15))

                recnos = np.unique(radar['recno'])
                for rec in recnos:
                    radar_plot = radar_old[radar_old['recno'] == rec]
                    plt.plot(radar_plot['ne'], radar_plot['gdalt'], label=np.unique(radar_plot['dates'])[0], alpha=0.25)

                radar_mean['nel'] = np.log10(radar_mean['ne'])

                radar_mean_window = radar_mean[((radar_mean.index <= 600) & (radar_mean.index >= 200))]


                # filter nan values
                radar_mean_window = radar_mean_window.dropna(subset=['ne'])

                plt.plot(radar_mean_notsmooth['ne'], radar_mean_notsmooth.index, color='gray', ls='--', alpha=0.75,
                         label='smoothed average profile')
                plt.plot(radar_mean['ne'], radar_mean.index, color='k', label='smoothed average profile')
                plt.title(radar_file.split('/')[-1])

                x = radar_mean_window['nel'].values
                y = radar_mean_window.index.values

                popt, _ = curve_fit(parabola, y, x, maxfev=1000000)
                a, b, c = popt

                # define a sequence of inputs between the smallest and largest known inputs
                x_line = np.arange(min(y), max(y), 1000)
                # calculate the output for the range
                y_line = parabola(x_line, a, b, c)

                plt.plot(10 ** y_line, x_line, '--', color='red', lw=2.5, label='curve fitting')

                plt.ylim(0, 1000)
                plt.legend()
                plt.savefig('../data/figures/f1f2_{n}.png'.format(n=radar_file.split('/')[-1]))
                print(radar_file.split('/')[-1])

        del radar
        del radar_mean
        del radar_mean_window
        del radar_mean_notsmooth
        del radar_old
        del radar_plot
        del x
        del y
        del x_line
        del y_line
        gc.collect()


    except:
        pass