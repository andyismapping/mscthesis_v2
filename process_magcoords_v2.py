import glob
import numpy as np
import pandas as pd
import pandas
import datetime
import math
import aacgmv2

import warnings
warnings.filterwarnings(action='once')

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

def geo2mag(latitude, longitude, altitude, timestamp):

    # mlat, mlon, mlt = aacgmv2.wrapper.get_aacgm_coord(latitude, longitude, altitude, timestamp, method='’ALLOWTRACE|GEOCENTRIC’')

    mlat, mlon, _ = aacgmv2.wrapper.convert_latlon(latitude, longitude, altitude, timestamp, method_code="G2A|ALLOWTRACE")

    if np.isnan(mlon):
        mlt = np.nan
    else:
        mlt = aacgmv2.wrapper.convert_mlt(mlon, timestamp, m2a=False)[0]

    return mlat, mlon, mlt

# GR
gf_files = glob.glob('../data/processed/Absolute_Ne_converted/*/GR_OPER_NE__KBR_2F_*.csv')
gf_files.sort()
for gf_file in gf_files:

    try:

        gf = pd.read_csv(gf_file).drop(['Unnamed: 0'], axis=1)
        gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])

        gf['Re'] = gf.Latitude.apply(lambda x: earth_radius(x))

        gf[['mlat','mlon', 'mlt']] = (gf.apply(lambda x: geo2mag(x['Latitude'], x['Longitude'], (x['Radius'] * 0.001 - x['Re']), x['Timestamp']), axis = 1).values.tolist())

        # gf = gf.dropna()
        #
        # gf['hour'] = gf.mlt.apply(lambda x : int(x))

        gf.to_csv('../data/processed/Absolute_Ne_v2/GRACE/{filename}'.format(filename=gf_file.split('/')[-1]))

    except:
        print(gf_file)
        pass

# GF
gf_files = glob.glob('../data/processed/Absolute_Ne_converted/*/GF_OPER_NE__KBR_2F_*.csv')
gf_files.sort()
for gf_file in gf_files:

    try:

        gf = pd.read_csv(gf_file).drop(['Unnamed: 0'], axis=1)
        gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])

        gf['Re'] = gf.Latitude.apply(lambda x: earth_radius(x))

        gf[['mlat','mlon', 'mlt']] = (gf.apply(lambda x: geo2mag(x['Latitude'], x['Longitude'], (x['Radius'] * 0.001 - x['Re']), x['Timestamp']), axis = 1).values.tolist())

        # gf = gf.dropna()
        #
        # gf['hour'] = gf.mlt.apply(lambda x : int(x))

        gf.to_csv('../data/processed/Absolute_Ne_v2/GRACEFO/{filename}'.format(filename=gf_file.split('/')[-1]))
    except:
        print(gf_file)
        pass