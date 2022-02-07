import glob
import numpy as np
import pandas as pd
from astropy.time import Time
import cdflib
import datetime
from astropy.table import Table
import math
from scipy.optimize import curve_fit


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


spatial_window = 5
temporal_window = 15
altitude_window = 100

radar_list = glob.glob('../data/external/Madrigal/2018-2021/*/*.hdf5')
radar_list.sort()

gf_list = glob.glob('../data/interim/GRACEFO/KBRNE_relative_v2/dat/*')
gf_list.sort()

headerlist = ['CDF Epoch', 'GPS', 'Latitude', 'Longitude', 'Radius', 'Latitude_QD', 'Longitude_QD',
              'MLT', 'GRACE_1_Position_0', 'GRACE_1_Position_1', 'GRACE_1_Position_2', 'GRACE_2_Position_1',
              'GRACE_2_Position_2', 'GRACE_2_Position_3', 'Iono_Corr', 'Distance',
              'Relative_Hor_TEC', 'Relative_Ne']

gf_list_datetimes = []
for gf_file in gf_list:
    gf = pd.read_csv(gf_file, sep='\s+', header=0, index_col=False, names=headerlist)
    gf_list_datetimes.append((Time(gf['CDF Epoch'].values[-1], format='cdf_epoch').datetime).strftime('%Y-%m-%d'))


def find_conjunctions(radar_file, gf_list, gf_list_datetimes):
    radar_select_0 = []
    gf_select_0 = []
    radar_select_1 = []
    gf_select_1 = []
    radar_select_2 = []
    gf_select_2 = []

    radar_metadata = []

    radar = (Table.read(radar_file, path='Data/Table Layout')).to_pandas()
    radar_date = datetime.datetime(int(radar.year[0]), int(radar.month[0]), int(radar.day[0])).strftime('%Y-%m-%d')

    date, idx_gf, idx_radar = np.intersect1d(gf_list_datetimes, radar_date, return_indices=True)

    if len(idx_radar) > 0:

        gf_file = gf_list[idx_gf[0]]
        gf = pd.read_csv(gf_file, sep='\s+', header=0, index_col=False, names=headerlist)
        gf['dates'] = (Time(gf['CDF Epoch'].values, format='cdf_epoch')).datetime

        radar_metadata = (Table.read(radar_file, path='Metadata/Experiment Parameters')).to_pandas()

        names = np.array([x.decode() for x in radar_metadata['name']])
        values = np.array([x.decode() for x in radar_metadata['value']])

        radar_metadata = pd.DataFrame({'name': names,
                                       'value': values})
        radar_lat = float(radar_metadata[radar_metadata['name'] == 'instrument latitude']['value'].values[0])
        radar_lon = float(radar_metadata[radar_metadata['name'] == 'instrument longitude']['value'].values[0])
        radar_lon = (((radar_lon + 180) % 360) - 180)
        radar['dates'] = Time(radar[['ut1_unix', 'ut2_unix']].mean(axis=1), format='unix').datetime

        gf = gf[((gf['Latitude'] <= radar_lat + spatial_window) & (gf['Latitude'] >= radar_lat - spatial_window))]
        gf = gf[((gf['Longitude'] <= radar_lon + spatial_window) & (gf['Longitude'] >= radar_lon - spatial_window))]

        if len(gf) > 0:

            gf_pass = gf[gf.diff(axis=0)['dates'] > datetime.timedelta(minutes=temporal_window)]

            if len(gf_pass) == 0:  # only one passage

                time_window_i = gf['dates'][gf.index.min()] - datetime.timedelta(minutes=temporal_window)
                time_window_f = gf['dates'][gf.index.max()] + datetime.timedelta(minutes=temporal_window)

                radar['dates'] = Time(radar[['ut1_unix', 'ut2_unix']].mean(axis=1), format='unix').datetime

                radar_select_0 = radar[(radar['dates'] >= time_window_i) & (radar['dates'] <= time_window_f)]
                gf_select_0 = gf[(gf['dates'] >= time_window_i) & (gf['dates'] <= time_window_f)]

                if len(radar_select_0) > 0:
                    Re = earth_radius(radar_lat)
                    altitude = np.mean(gf_select_0['Radius'] * 0.001 - Re)

                    altitude_window_i = altitude - altitude_window
                    altitude_window_f = altitude + altitude_window

                    radar_select_0 = radar_select_0[
                        (radar_select_0['gdalt'] >= altitude_window_i) & (radar_select_0['gdalt'] <= altitude_window_f)]

            if len(gf_pass) == 1:  # two passages

                pass_idx = gf[gf['dates'] == gf_pass['dates'][gf_pass.index.min()]].index[0]

                # first one

                # define time window
                time_window_i = gf['dates'][gf.index.min()] - datetime.timedelta(minutes=15)
                time_window_f = gf['dates'][
                                    gf.index[np.argwhere(gf.index == pass_idx)[0][0] - 1]] + datetime.timedelta(
                    minutes=15)

                radar_select_1 = radar[(radar['dates'] >= time_window_i) & (radar['dates'] <= time_window_f)]
                gf_select_1 = gf[(gf['dates'] >= time_window_i) & (gf['dates'] <= time_window_f)]

                if len(radar_select_1) > 0:
                    Re = earth_radius(radar_lat)
                    altitude = np.mean(gf_select_1['Radius'] * 0.001 - Re)

                    altitude_window_i = altitude - altitude_window
                    altitude_window_f = altitude + altitude_window

                    radar_select_1 = radar_select_1[
                        (radar_select_1['gdalt'] >= altitude_window_i) & (radar_select_1['gdalt'] <= altitude_window_f)]

                # second one

                # define time window
                time_window_i = gf['dates'][gf.index[np.argwhere(gf.index == pass_idx)[0][0]]] - datetime.timedelta(
                    minutes=15)
                time_window_f = gf['dates'][gf.index.max()] + datetime.timedelta(minutes=15)

                radar_select_2 = radar[(radar['dates'] >= time_window_i) & (radar['dates'] <= time_window_f)]
                gf_select_2 = gf[(gf['dates'] >= time_window_i) & (gf['dates'] <= time_window_f)]

                if len(radar_select_2) > 0:
                    Re = earth_radius(radar_lat)
                    altitude = np.mean(gf_select_2['Radius'] * 0.001 - Re)  # 0.001 convert m to km

                    altitude_window_i = altitude - altitude_window
                    altitude_window_f = altitude + altitude_window

                    radar_select_2 = radar_select_2[
                        (radar_select_2['gdalt'] >= altitude_window_i) & (radar_select_2['gdalt'] <= altitude_window_f)]

    del radar
    del gf

    return radar_select_0, gf_select_0, radar_select_1, gf_select_1, radar_select_2, gf_select_2, radar_metadata, Re


def calc_conjunctions(radar_select, gf_select, Re, r):
    radar_select['nel'] = np.log10(radar_select['ne'])

    radar_mean = radar_select.groupby(['gdalt']).mean()

    gf_alt = np.mean(gf_select['Radius'] * 0.001 - Re)

    x = radar_mean['nel'].values
    y = radar_mean.index.values

    # curve fit
    popt, _ = curve_fit(objective, x, y, maxfev=10000)
    a, b = popt

    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), 10e-6)
    # calculate the output for the range
    y_line = objective(x_line, a, b)

    idx = find_nearest_idx(y_line, gf_alt)
    radar_alt = y_line[idx]
    radar_nel = x_line[idx]

    radar_recno = len(np.unique(radar_select['recno']))

    gf_ne = np.mean(gf_select['Relative_Ne'])
    return radar_nel, radar_alt, radar_recno, gf_ne


radars = glob.glob('../data/external/Madrigal/2018-2021/*')

list_cdfepoch = []
list_radar = []
list_radarNe = []
list_radarProfiles = []
list_gfNe = []

for r in iter(range(0, len(radars))):

    print('------')
    print(r, radars[r])
    radar_list = (glob.glob('../data/external/Madrigal/2018-2021/' + radars[r].split('/')[-1] + '/*'))
    names = []
    count = 0

    for i in iter(range(0, len(radar_list))):

        try:
            radar_select_0, gf_select_0, radar_select_1, gf_select_1, radar_select_2, gf_select_2, radar_metadata, Re = find_conjunctions(
                radar_file=radar_list[i], gf_list=gf_list, gf_list_datetimes=gf_list_datetimes)

            if (len(radar_select_0) > 0) & (len(gf_select_0) > 0):
                radar_nel, radar_alt, radar_recno, gf_ne = calc_conjunctions(radar_select_0, gf_select_0, Re, r)

                list_cdfepoch.append(gf_select_0['CDF Epoch'].median())
                list_radar.append(radars[r].split('/')[-1])
                list_radarNe.append(10 ** radar_nel)
                list_radarProfiles.append(radar_recno)
                list_gfNe.append(gf_ne)

                count = count + 1

            if (len(radar_select_1) > 0) & (len(gf_select_1) > 0):
                radar_nel, radar_alt, radar_recno, gf_ne = calc_conjunctions(radar_select_1, gf_select_1, Re, r)

                list_cdfepoch.append(gf_select_1['CDF Epoch'].median())
                list_radar.append(radars[r].split('/')[-1])
                list_radarNe.append(10 ** radar_nel)
                list_radarProfiles.append(radar_recno)
                list_gfNe.append(gf_ne)

                count = count + 1

            if (len(radar_select_2) > 0) & (len(gf_select_2) > 0):
                radar_nel, radar_alt, radar_recno, gf_ne = calc_conjunctions(radar_select_2, gf_select_2, Re, r)

                list_cdfepoch.append(gf_select_2['CDF Epoch'].median())
                list_radar.append(radars[r].split('/')[-1])
                list_radarNe.append(10 ** radar_nel)
                list_radarProfiles.append(radar_recno)
                list_gfNe.append(gf_ne)

                count = count + 1

        except:
            pass

    print(count)
    print(len(radar_list))

radar_calib = pd.DataFrame(data={'cdf-epoch': list_cdfepoch,
                                 'radar': list_radar,
                                 'radar_ne': list_radarNe,
                                 'radar_nprofiles': list_radarProfiles,
                                 'gf_ne': list_gfNe})

radar_calib.to_csv('../data/processed/RADAR_CALIB_TEST.csv')
