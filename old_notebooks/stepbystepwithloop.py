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

def parabola(x, a, b, c):
    return a*x**2 + b*x + c


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
radar_list = glob.glob('../data/external/Madrigal/MillstoneHillISRadar/*.hdf5')

radar_list.sort()

# gf_ts = pd.read_csv('../data/processed/GF_ts_complete.csv')
# gf_ts['dates'] = (Time(gf_ts['CDF Epoch'].values, format='cdf_epoch')).datetime

headerlist = ['CDF Epoch', 'GPS', 'Latitude', 'Longitude', 'Radius', 'Latitude_QD', 'Longitude_QD',
              'MLT GRACE_1_Position', 'GRACE_2_Position', 'Iono_Corr', 'Distance', 'Relative_Hor_TEC', 'Relative_Ne']

list_cdfepoch = []
list_radar = []
list_radarNe = []
list_radarProfiles = []
list_gfNe = []

for radar_file in radar_list:
        # print(radar_file.split('/')[-1])

        try:

            radar = (Table.read(radar_file, path='Data/Table Layout')).to_pandas()
            radar['dates'] = Time(radar[['ut1_unix', 'ut2_unix']].mean(axis=1), format='unix').datetime

            gf_file = glob.glob('../data/interim/GRACE/dat/GR_OPER_NE__KBR_2F_{date}*.dat'.format(
                date=radar['dates'][radar.index[0]].strftime('%Y%d%m')))[0]

            gf = pd.read_csv(gf_file, sep='\s+', header=0, index_col=False, names=headerlist)
            gf['dates'] = (Time(gf['CDF Epoch'].values, format='cdf_epoch')).datetime

            radar_metadata = (Table.read(radar_file, path='Metadata/Experiment Parameters')).to_pandas()

            names = np.array([x.decode() for x in radar_metadata['name']])
            values = np.array([x.decode() for x in radar_metadata['value']])

            radar_metadata = pd.DataFrame({'name': names,
                                           'value': values})
            radar_metadata.set_index('name', inplace=True)
            radar_metadata = dict(pd.DataFrame.transpose(radar_metadata))

            radar_lat = float(radar_metadata['instrument latitude'].value)
            radar_lon = float(radar_metadata['instrument longitude'].value)
            radar_lon = (((radar_lon + 180) % 360) - 180)

            # filter space
            gf = gf[((gf['Latitude'] <= radar_lat + spatial_window) & (gf['Latitude'] >= radar_lat - spatial_window))]
            gf = gf[((gf['Longitude'] <= radar_lon + spatial_window) & (gf['Longitude'] >= radar_lon - spatial_window))]

            # filter time

            gf = gf[((gf['dates'] <= radar['dates'].max() + datetime.timedelta(hours=6)) & (
                            gf['dates'] >= radar['dates'].min() - datetime.timedelta(hours=6)))]

            radar = radar[((radar['dates'] <= gf['dates'].max() + datetime.timedelta(minutes=temporal_window)) & (
                            radar['dates'] >= gf['dates'].min() - datetime.timedelta(minutes=temporal_window)))]

            # save original
            radar_old = radar

            # black profile

            radar_mean = pd.DataFrame({'gdalt': radar['gdalt'],
                                       'ne': radar['ne']})

            radar_mean = radar_mean.groupby(['gdalt']).mean()

            Xresampled = np.arange(int(radar_mean.index.min()), int(radar_mean.index.max()), 10)

            radar_mean = radar_mean.reindex(radar_mean.index.union(Xresampled)).interpolate('spline', order=1).loc[
                Xresampled]

            # filter altitude
            Re = earth_radius(radar_lat)
            gf_alt = np.mean(gf['Radius'] * 0.001 - Re)

            radar = radar[
                ((radar['gdalt'] <= gf_alt + altitude_window) & (radar['gdalt'] >= gf_alt - altitude_window))]

            if len(radar) > 0:

                print(radar_file.split('/')[-1])
                print('GF', gf.dates.min(), gf.dates.max())
                print('RADAR', radar.dates.min(), radar.dates.max())
                print('--------------')
        # except:
        #     pass

                fig, ax = plt.subplots(figsize=(10, 12.5))

                recnos = np.unique(radar['recno'])
                #         for rec in recnos:
                #             radar_plot = radar_old[radar_old['recno'] == rec]
                #             plt.plot(radar_plot['ne'], radar_plot['gdalt'] , label = np.unique(radar_plot['dates'])[0], alpha = 0.25)

                #         radar_mean_window = radar_mean[((radar_mean.index <= 600) & (radar_mean.index >= 200))]
                radar_mean_window = radar_mean[((radar_mean.index <= gf_alt + altitude_window) & (
                            radar_mean.index >= gf_alt - altitude_window))]

                radar['nel'] = np.log10(radar['ne'])
                radar_mean['nel'] = np.log10(radar_mean['ne'])
                radar_old['nel'] = np.log10(radar_old['ne'])
                radar_mean_window['nel'] = np.log10(radar_mean_window['ne'])

                # filter nan values
                radar_mean = radar_mean.dropna(subset=['nel'])
                radar_mean_window = radar_mean_window.dropna(subset=['nel'])

                plt.plot(radar_mean['ne'], radar_mean.index, color='k', label='Radar smoothed average profile')
                #         plt.title(radar_file.split('/')[-1])

                x = radar_mean_window['nel'].values
                y = radar_mean_window.index.values

                popt, _ = curve_fit(parabola, y, x, maxfev=10000)
                a, b, c = popt

                # define a sequence of inputs between the smallest and largest known inputs
                x_line = np.arange(min(y), max(y), 10e-6)
                # calculate the output for the range
                y_line = parabola(x_line, a, b, c)

                plt.plot(10 ** y_line, x_line, '--', color='red', lw=2.5, label='Radar curve fit')

                gf_ne = np.mean(gf['Relative_Ne'])
                plt.scatter(gf_ne, gf_alt, marker='X', s=60, color='b', label='GF average')

                idx = find_nearest_idx(x_line, gf_alt)
                radar_nel = y_line[idx]
                radar_alt = x_line[idx]
                plt.scatter(10 ** radar_nel, radar_alt, marker='X', s=60, color='r', label='Radar extrapolated Ne')

                plt.xlabel('Electron density [m-3]', fontsize='medium')
                plt.ylabel('Altitude [km]', fontsize='medium')

                #         plt.axhline(y=gf_alt, color='gray', linestyle='-')
                plt.title('Curve fitting for Millstone Hill ISR \n 2018-06-14', fontsize='large')

                plt.axhline(y=gf_alt, color='gray', linestyle='--', alpha=0.5)

                plt.legend(loc='upper right')
                plt.show()
                pass

                list_cdfepoch.append(gf['CDF Epoch'].median())
                list_radar.append(radar_metadata['instrument'].value)
                list_radarNe.append(10 ** radar_nel)
                list_radarProfiles.append(recnos)
                list_gfNe.append(gf_ne)

                print(radar_nel, gf_ne)
                del radar
                del gf
                del x
                del y
                gc.collect()

        # else:
        #     #             print('no gf file on: ', radar['dates'][radar.index[0]].strftime('%Y%d%m')
        #     pass

        except:
            pass

radar_calib = pd.DataFrame(data={'cdf-epoch': list_cdfepoch,
                                 'radar': list_radar,
                                 'radar_ne': list_radarNe,
                                 'radar_nprofiles': list_radarProfiles,
                                 'gf_ne': list_gfNe})

radar_calib.to_csv('RADAR_CALIB_TEST.csv')