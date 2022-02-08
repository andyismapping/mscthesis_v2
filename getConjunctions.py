# import modules
import glob
import numpy as np
import pandas
import pandas as pd
from astropy.time import Time
import datetime
from astropy.table import Table
import math
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from spacepy import pycdf
import datetime as dt
import warnings
import os

warnings.filterwarnings(action='once')


# defined functions
def earth_radius(B):
    """
    Function to calculate earth radius

    :param B: latitude in degrees
    :return:
    """
    B = math.radians(B)  # converting into radians
    a = 6378.137  # Radius at sea level at equator
    b = 6356.752  # Radius at poles
    c = (a ** 2 * math.cos(B)) ** 2
    d = (b ** 2 * math.sin(B)) ** 2
    e = (a * math.cos(B)) ** 2
    f = (b * math.sin(B)) ** 2
    R = math.sqrt((c + d) / (e + f))
    return R


def parabola(x, a, b, c):
    """
    Parabola function

    :param x: nel
    :param a: curve_fit a parameter
    :param b: curve_fit b parameter
    :param c: curve_fit c parameter
    :return:
    """
    return a * x ** 2 + b * x + c


def find_nearest_idx(array, value):
    """
    Function to find the nearest value in an array

    :param array: x array (radar altitude array)
    :param value: y value (gf altitude value)
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# defined parameters
spatial_window = 5  # km
temporal_window = 15  # min
altitude_window = 100  # km
altitude_window2 = 20  # km


def open_radar(radar_file):
    """
    Function to open Madrigal radar files

    :param radar_file: radar file name with path
    :return:
    """
    radar = (Table.read(radar_file, path='Data/Table Layout')).to_pandas()
    radar['dates'] = Time(radar[['ut1_unix', 'ut2_unix']].mean(axis=1), format='unix').datetime

    return radar


var_set1 = ['gdalt', 'ne']
var_set2 = ['gdalt', 'nel']
var_set3 = ['range', 'ne']
var_set4 = ['range', 'nel']


def get_var_set(radar):
    """
    Function to determine the existence of altitude and electron density variables in a radar df. Possible combinations
    are described in var_set1, var_set2, var_set3, and var_set4

    :param radar: radar df
    :return:
    """
    for var_set in [var_set1, var_set2, var_set3, var_set4]:

        if all([l in radar.columns for l in var_set]):
            return var_set


def set_vars(radar, variables):
    """
    Function to create ne when only nel exists

    :param radar: radar df
    :param vars: var set
    :return:
    """
    if variables[1] == 'nel':
        radar['ne'] = 10 ** radar['nel']
    else:
        pass

    return radar


def find_gf_file(radar):
    """
    Function to find GRACE/GRACE-FO file matching the mean date of the radar file
    Filename starting with GR is for GRACE, and GF for GRACEFO, for now it's selecting all.

    :param radar: radar df
    :return:
    """
    gf_file = glob.glob('../data/interim/Absolute_Ne/*/*_OPER_NE__KBR_2F_{date}*.cdf'.format(
        date=radar['dates'].mean().strftime('%Y%m%d')))[0]
    return gf_file


def open_radar_metadata(radar_file):
    """
    Function to open the radar metadata information and convert it to a df

    :param radar_file: radar file name with path
    :return:
    """
    radar_metadata = (Table.read(radar_file, path='Metadata/Experiment Parameters')).to_pandas()
    names = np.array([x.decode() for x in radar_metadata['name']])
    values = np.array([x.decode() for x in radar_metadata['value']])

    radar_metadata = pd.DataFrame({'name': names,
                                   'value': values})
    radar_metadata.set_index('name', inplace=True)
    radar_metadata = dict(pd.DataFrame.transpose(radar_metadata))

    return radar_metadata


def get_radar_metadata_lat_lon(radar_metadata):
    """
    Function to extract latitude and longitude from the radar metadata df. Longitudes are converted to the -180 180
    format

    :param radar_metadata: radar metadata df
    :return:
    """
    radar_lat = float(radar_metadata['instrument latitude'].value)
    radar_lon = float(radar_metadata['instrument longitude'].value)
    radar_lon = (((radar_lon + 180) % 360) - 180)

    return radar_lat, radar_lon


def convert_gf_to_pandas(gf):
    """
    Function to convert gf hdf5 into a df. LEO position parameters are removed from the final df

    :param gf: gf df
    :return:
    """
    keys = list(gf.keys())
    keys.remove("LEO_Position")
    gf_dict = {k: gf[k][...] for k in keys}

    gf = pandas.DataFrame(gf_dict)

    return gf


def filter_lat_lon_window(gf, radar_lat, radar_lon, spatial_window):
    """
    Function to filter gf observations measured only in the vicinity of the radar observatory, a spatial window of pre
    determined km value is specified earlier in the code as 'spatial_window'

    :param gf: gf df
    :param radar_lat: radar latitude in degrees
    :param radar_lon: radar longitude in degrees
    :param spatial_window: defined parameter in km
    :return:
    """
    gf = gf[((gf['Latitude'][...] <= radar_lat + spatial_window) & (gf['Latitude'][...] >= radar_lat - spatial_window))]
    gf = gf[
        ((gf['Longitude'][...] <= radar_lon + spatial_window) & (gf['Longitude'][...] >= radar_lon - spatial_window))]

    return gf


def filter_time_window(gf, radar, temporal_window):
    """
    Function to filter both gf and radar observations measured within a pre determined time window in minutes specified
    earlier in the code as 'time_window'.
    Since GRACE/GRACE-FO can pass through the same coordinates twice a day the function looks for a gap in observations,
    if the gap is bigger than 6 hours then each segment is treated individually, allowing for two conjunctions to happen
    in the same day. The outuput is divided in 3 sets of filtered radar and gf datasets, 0 is for when only one
    conjunction can happens (no gaps in the gf df), 1 and 2 are for each segment when a gap in gf df is identified. The
    function alaways returns all 3 sets.
    gf df contains only the observations within the spatial window, radar df contains observations measured within gf
    df observations +/- the time window.

    :param gf: gf df
    :param radar: radar df
    :param temporal_window: defined parameter in minutes
    :return:
    """
    gf_time_window_max = gf['Timestamp'].max() + datetime.timedelta(minutes=temporal_window)
    gf_time_window_min = gf['Timestamp'].min() - datetime.timedelta(minutes=temporal_window)
    gf_time_window_diff = (gf_time_window_max - gf_time_window_min).total_seconds() / 3600

    if gf_time_window_diff < 6:
        radar_0 = radar[((radar['dates'] <= gf_time_window_max) & (radar['dates'] >= gf_time_window_min))]
        gf_0 = gf[((gf['Timestamp'] <= gf_time_window_max) & (gf['Timestamp'] >= gf_time_window_min))]

        gf_1 = pd.DataFrame()
        radar_1 = pd.DataFrame()
        gf_2 = pd.DataFrame()
        radar_2 = pd.DataFrame()

    else:
        deltas = gf['Timestamp'].diff()
        gap = deltas[deltas > dt.timedelta(hours=6)]

        gf_1 = gf[gf.index < gap.index.values[0]]
        gf_1_time_window_max = gf_1['Timestamp'].max() + datetime.timedelta(minutes=temporal_window)
        gf_1_time_window_min = gf_1['Timestamp'].min() - datetime.timedelta(minutes=temporal_window)

        radar_1 = radar[((radar['dates'] <= gf_1_time_window_max) & (radar['dates'] >= gf_1_time_window_min))]
        gf_1 = gf_1[((gf_1['Timestamp'] <= gf_1_time_window_max) & (gf_1['Timestamp'] >= gf_1_time_window_min))]

        gf_2 = gf[gf.index >= gap.index.values[0]]
        gf_2_time_window_max = gf_2['Timestamp'].max() + datetime.timedelta(minutes=temporal_window)
        gf_2_time_window_min = gf_2['Timestamp'].min() - datetime.timedelta(minutes=temporal_window)

        radar_2 = radar[((radar['dates'] <= gf_2_time_window_max) & (radar['dates'] >= gf_2_time_window_min))]
        gf_2 = gf_2[((gf_2['Timestamp'] <= gf_2_time_window_max) & (gf_2['Timestamp'] >= gf_2_time_window_min))]

        gf_0 = pd.DataFrame()
        radar_0 = pd.DataFrame()

    return radar_0, gf_0, radar_1, gf_1, radar_2, gf_2


def radar_avg_profile(radar, variables):
    """
    Function to resample radar df in a 10 km altitude regular time series using a spline interpolation

    :param radar: radar df
    :param vars: var set
    :return:
    """
    radar_mean = pd.DataFrame({'{}'.format(variables[0]): radar['{}'.format(variables[0])],
                               'ne': radar['ne']})

    radar_mean = radar_mean.groupby(['{}'.format(variables[0])]).mean()

    Xresampled = np.arange(int(radar_mean.index.min()), int(radar_mean.index.max()), 10)

    radar_mean = radar_mean.reindex(radar_mean.index.union(Xresampled)).interpolate('spline', order=1).loc[
        Xresampled]

    return radar_mean


def get_gf_alt(gf, Re):
    """
    Function to convert gf radius to altitude

    :param gf: gf df
    :param Re:  earth radius in km
    :return:
    """
    gf_alt = np.mean(gf['Radius'] * 0.001 - Re)
    return gf_alt


def filter_altitude(radar, gf_alt, variables, altitude_window):
    """
    Function to filter radar df within the gf altitude +/- a pre defined altitude window.

    :param radar: radar df
    :param gf_alt: gf altitude in km
    :param vars: var set
    :param altitude_window: defined parameter in km
    :return:
    """
    radar = radar[
        ((radar['{}'.format(variables[0])] <= gf_alt + altitude_window) & (
                radar['{}'.format(variables[0])] >= gf_alt - altitude_window))]

    return radar


def plot_ne_altitude(recnos, radar_old, radar_mean, y_line, x_line, gf_nel, gf_alt, radar_nel, radar_alt):
    """
    Function to plot the all radar profiles as colorful lines, the averaged radar profile in black, the curve fit in a
    dashed red line, the radar conjunction value as a red mark, and the gf conjunction as a blue mark

    :param recnos: radar profiles
    :param radar_old: original radar df
    :param radar_mean: mean radar df
    :param y_line: parabola fit for radar nel
    :param x_line: parabola fit for radar altitude
    :param gf_ne: gf average ne
    :param gf_alt: gf altitude
    :param radar_nel: radar nel at gf altitude
    :param radar_alt: radar altitude
    :return:
    """
    for rec in recnos:
        radar_plot = radar_old[radar_old['recno'] == rec]
        plt.plot(radar_plot['nel'], radar_plot['{}'.format(variables[0])],
                 alpha=0.25)  # label = np.unique(radar_plot['dates'])[0]

    plt.plot(radar_mean['nel'], radar_mean.index, color='k', label='Radar smoothed average profile')
    plt.plot(y_line, x_line, '--', color='red', lw=2.5, label='Radar curve fit')


    plt.scatter(gf_nel, gf_alt, marker='X', s=60, color='b', label='GF average')
    plt.scatter(radar_nel, radar_alt, marker='X', s=60, color='r', label='Radar extrapolated Ne')

    plt.xlabel('Log Electron density [$m^{-3}$]', fontsize='medium')
    plt.ylabel('Altitude [km]', fontsize='medium')

    plt.axhline(y=gf_alt, color='gray', linestyle='-')
    plt.legend(loc='lower right')
    # plt.show()

# radar files
radar_list = glob.glob('../data/external/Madrigal/madrigal/*/*.hdf5')
radar_list_pokerflat = glob.glob('../data/external/Madrigal/madrigal/PokerFlat/*/*.hdf5')
radar_list_pokerflat_2017 = glob.glob('../data/external/Madrigal/madrigal/PokerFlat/2017/*/*.hdf5')
# radar_list_north = glob.glob('../data/external/Madrigal/madrigal/ResoluteBayNorthISRadar/*.hdf5')
# radar_list_canada = glob.glob('../data/external/Madrigal/madrigal/ResoluteBayCanadaISRadar/*.hdf5')


radar_list = radar_list + radar_list_pokerflat + radar_list_pokerflat_2017
# radar_list = radar_list_north + radar_list_canada

radar_list.sort()


for radar_file in radar_list:
    # print(radar_file.split('/')[-1])

    try:

        # open radar
        radar = open_radar(radar_file)
        variables = get_var_set(radar)
        radar = set_vars(radar, variables)

        # open gf
        gf_file = find_gf_file(radar)
        gf = pycdf.CDF(gf_file)

        # open radar metadata
        radar_metadata = open_radar_metadata(radar_file)
        radar_lat, radar_lon = get_radar_metadata_lat_lon(radar_metadata)

        # convert gf to pandas
        gf = convert_gf_to_pandas(gf)

        # filter lat lon
        gf = filter_lat_lon_window(gf, radar_lat, radar_lon, spatial_window)

        # filter time
        radar_0, gf_0, radar_1, gf_1, radar_2, gf_2 = filter_time_window(gf, radar, temporal_window)

        df_set0 = [radar_0, gf_0]
        df_set1 = [radar_1, gf_1]
        df_set2 = [radar_2, gf_2]

        for df_set in [df_set0, df_set1, df_set2]:
            radar = df_set[0]
            gf = df_set[1]

            if (len(radar) > 0) & (len(gf) > 0):

                # save original for plotting later
                radar_old = radar

                # average profile
                radar_mean = radar_avg_profile(radar, variables)

                # filter altitude
                Re = earth_radius(radar_lat)
                gf_alt = get_gf_alt(gf, Re)

                radar = filter_altitude(radar, gf_alt, variables, altitude_window)
                radar2 = filter_altitude(radar, gf_alt, variables, altitude_window2)

                if len(radar2) > 0:
                    print(radar_file.split('/')[-1])
                    print('GF', gf.Timestamp.min(), gf.Timestamp.max())
                    print('RADAR', radar.dates.min(), radar.dates.max())
                    print('--------------')

                    # numerical operations
                    recnos = np.unique(radar['recno'])

                    radar_mean_window = radar_mean[((radar_mean.index <= gf_alt + altitude_window) & (
                            radar_mean.index >= gf_alt - altitude_window))]

                    radar['nel'] = np.log10(radar['ne'])
                    radar_mean['nel'] = np.log10(radar_mean['ne'])
                    radar_old['nel'] = np.log10(radar_old['ne'])
                    radar_mean_window['nel'] = np.log10(radar_mean_window['ne'])

                    # filter nan values
                    radar_mean = radar_mean.dropna(subset=['nel'])
                    radar_mean_window = radar_mean_window.dropna(subset=['nel'])

                    # curve fit
                    x = radar_mean_window['nel'].values
                    y = radar_mean_window.index.values

                    popt, _ = curve_fit(parabola, y, x, maxfev=10000)
                    a, b, c = popt

                    x_line = np.arange(min(y), max(y), 10e-6)
                    y_line = parabola(x_line, a, b, c)

                    gf_nel = np.log10(np.mean(gf['Absolute_Ne']))

                    # get the closest point to GF altitude
                    idx = find_nearest_idx(x_line, gf_alt)
                    radar_nel = y_line[idx]
                    radar_alt = x_line[idx]

                    # plot
                    fig, ax = plt.subplots(figsize=(10, 12.5))

                    plot_ne_altitude(recnos, radar_old, radar_mean, y_line, x_line, gf_nel, gf_alt, radar_nel, radar_alt)

                    plt.title(radar_file.split('/')[-1], fontsize='large')
                    plt.savefig("../figures/v2/{name}.png".format(name=radar_file.split('/')[-1]))
                    plt.close()

                    # save output
                    list_datetime = []
                    list_radar = []
                    list_NelRadar = []
                    list_nProfiles = []
                    list_NelGf = []
                    list_filenameRadar = []
                    list_NelDiff = []
                    list_mission = []

                    list_datetime.append(gf['Timestamp'].median())
                    list_radar.append(radar_metadata['instrument'].value)
                    list_NelRadar.append(radar_nel)
                    list_nProfiles.append(len(recnos))
                    list_NelGf.append(gf_nel)
                    list_filenameRadar.append(radar_file.split('/')[-1])
                    list_NelDiff.append(gf_nel - radar_nel)
                    list_mission.append(gf_file.split('/')[-1][0:2])

                    radar_calib = pd.DataFrame(data={'date': list_datetime,
                                                     'radar': list_radar,
                                                     'radar_file': list_filenameRadar,
                                                     'nel_radar': list_NelRadar,
                                                     'n_profiles': list_nProfiles,
                                                     'nel_gf': list_NelGf,
                                                     'nel_diff': list_NelDiff,
                                                     'mission': list_mission})

                    output = '../tables/v2/conjunctions.csv'
                    radar_calib.to_csv(output, mode='a', header=not os.path.exists(output), index=False)

                    print(gf_nel, radar_nel)
                    print('--------------')

                    pass

    except:
        pass