import glob
import numpy as np
import pandas as pd
from astropy.time import Time
import cdflib
import datetime
import spacepy.datamodel as dm
from astropy.table import Table
import math
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib.dates as mdates


gf = pd.read_csv('../data/processed/GF_ts_complete.csv')
gf['dates'] = (Time(gf['CDF Epoch'].values, format='cdf_epoch')).datetime

radar_bias = pd.read_csv('../data/processed/RADAR_CALIB_TEST.csv')
radar_bias['dates'] = (Time(radar_bias['cdf-epoch'].values, format='cdf_epoch')).datetime

gaps = pd.read_csv('../data/interim/GRACEFO/KBRNE_relative_v2/aux_files/GRACE-FO-GAP-BIAS.ARC',sep='\s+')
gaps = gaps[gaps['new_ARC'] == 1]
gaps = gaps.reset_index()


arc = []
radar_cdf_epoch = []
radar_name = []
radar_profiles = []
radar_ne = []
gf_ne = []

# create df with arc number
for i in iter(range(0, len(gaps))):
    try:
        idx = gf[gf['CDF Epoch'] == gaps['CDF_Epoch'][i]].index
        idy = gf[gf['CDF Epoch'] == gaps['CDF_Epoch'][i + 1]].index - 1

        radar_bias_mean = radar_bias[(radar_bias['cdf-epoch'] >= gaps['CDF_Epoch'][i]) & (
                    radar_bias['cdf-epoch'] < gaps['CDF_Epoch'][i + 1])]

        if len(radar_bias_mean) > 0:

            for j in iter(range(0, len(radar_bias_mean))):
                arc.append(i)
                radar_cdf_epoch.append(radar_bias_mean['cdf-epoch'][radar_bias_mean.index[j]])
                radar_name.append(radar_bias_mean['radar'][radar_bias_mean.index[j]])
                radar_profiles.append(radar_bias_mean['radar_nprofiles'][radar_bias_mean.index[j]])
                radar_ne.append(radar_bias_mean['radar_ne'][radar_bias_mean.index[j]])
                gf_ne.append(radar_bias_mean['gf_ne'][radar_bias_mean.index[j]])

    except:
        pass

GF_RADAR_CALIB = pd.DataFrame({'arc': arc,
                               'radar_cdf_epoch': radar_cdf_epoch,
                               'radar_name': radar_name,
                               'radar_profiles': radar_profiles,
                               'radar_ne': radar_ne,
                               'gf_ne': gf_ne})

# offset for each individual radar
GF_RADAR_CALIB['mean_offset'] = GF_RADAR_CALIB['radar_ne'] - GF_RADAR_CALIB['gf_ne']

# offset by arc
arcs_offset = GF_RADAR_CALIB.groupby(["arc"])["mean_offset"].mean()
GF_RADAR_CALIB['arc_offset'] = GF_RADAR_CALIB['mean_offset']

for arc in GF_RADAR_CALIB['arc']:
    GF_RADAR_CALIB['arc_offset'][arc] = arcs_offset[arc]

# save csv
GF_RADAR_CALIB.to_csv('../data/processed/GF_RADAR_CALIB.csv')