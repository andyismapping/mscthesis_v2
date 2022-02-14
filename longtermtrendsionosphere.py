import glob
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib as mpl
import os
from scipy.stats import kde


import warnings
import string

warnings.filterwarnings(action='once')
# mpl.rcParams.update({'font.size': 16})

def process_heatmap(mission, method, level):
    # f107 = pd.read_csv('../data/external/f107/Kp_ap_Ap_SN_F107_since_1932.txt', delimiter=r"\s+", comment='#')
    # f107['date_simplified'] = f107.apply(lambda x: datetime.datetime(int(x['YYYY']), int(x['MM']), int(x['DD'])),
    #                                      axis=1)

    f107 = pd.read_fwf('../data/external/f107/omni2_daily_bOKKAhnXtE.lst', names=['year', 'day', 'hour', 'f10.7_index'])
    f107['date_simplified'] = f107.apply(
        lambda x: datetime.datetime(int(x.year), 1, 1) + datetime.timedelta(int(x.day) - 1), axis=1)
    f107['f10.7_index'][f107['f10.7_index'] > 400] = np.nan

    if level == 'low':
        f107 = f107[(f107['f10.7_index'] < 80)]
    elif level == 'medium':
        f107 = f107[(f107['f10.7_index'] >= 80) & (f107['f10.7_index'] < 120)]
    elif level == 'high':
        f107 = f107[(f107['f10.7_index'] >= 120)]
    else:
        print('wrong level')

    if mission == 'GR':
        f107 = f107[(f107['year'] >= 2002) & (f107['year'] <= 2017)]
    elif mission == 'GF':
        f107 = f107[(f107['year'] >= 2018) & (f107['year'] <= 2020)]
    else:
        print('wrong mission')

    f107 = f107.reset_index(drop=True)

    df_plot = pd.DataFrame()

    for i in range(0, len(f107)):

        date = f107['date_simplified'][i]

        try:

            df_file = glob.glob(
                '../data/processed/Absolute_Ne_v2/*/{mission}_OPER_NE__KBR_2F_{date}*.cdf.csv'.format(mission=mission,
                                                                                                   date=date.strftime(
                                                                                                       '%Y%m%d')))
            df = pd.read_csv(df_file[0], usecols=['mlat', 'mlt', 'Absolute_Ne'])
            df_plot = df_plot.append(df)

        except:
            pass

    yi = np.arange(-60, 60, 0.1)
    xi = np.arange(0, 24, 0.1)
    xi, yi = np.meshgrid(xi, yi)

    df_plot = df_plot.dropna()

    zi = griddata((df_plot['mlt'], df_plot['mlat']), df_plot['Absolute_Ne'], (xi, yi), method=method)
    np.save("../tables/v2/{mission}_{level}_{method}_mlat.npy".format(mission=mission, level=level, method=method), zi)


# process_heatmap(mission='GR', method='nearest', level='high')
# process_heatmap(mission='GR', method='nearest', level='medium')
# process_heatmap(mission='GR', method='nearest', level='low')
# # process_heatmap(mission='GF', method='nearest', level='high')
# process_heatmap(mission='GF', method='nearest', level='medium')
# process_heatmap(mission='GF', method='nearest', level='low')

def plot_heatmap(mission):
    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    y = np.arange(-60, 60, 0.1)
    x = np.arange(0, 24, 0.1)
    xi, yi = np.meshgrid(x, y)

    g = Gaussian2DKernel(5, 10)

    try:
        Z_low = np.load('../tables/v2/{mission}_low_nearest_mlat.npy'.format(mission=mission))
        Zi_low = convolve(Z_low, g)
    except:
        print('Z_low not found')
        Z_low = np.full(xi.shape, np.nan)
        Zi_low = np.full(xi.shape, np.nan)

    try:
        Z_medium = np.load('../tables/v2/{mission}_medium_nearest_mlat.npy'.format(mission=mission))
        Zi_medium = convolve(Z_medium, g)
    except:
        print('Z_medium not found')
        Z_medium = np.full(xi.shape, np.nan)
        Zi_medium = np.full(xi.shape, np.nan)

    try:
        Z_high = np.load('../tables/v2/{mission}_high_nearest_mlat.npy'.format(mission=mission))
        Zi_high = convolve(Z_high, g)
    except:
        print('Z_high not found')
        Z_high = np.full(xi.shape, np.nan)
        Zi_high = np.full(xi.shape, np.nan)
        pass

    levels = np.arange(10, 12.75, 0.25)


    fig, axs = plt.subplots(3, figsize=(12, 8), sharex=True, sharey=True)


    high = axs[0].contourf(xi, yi, np.log10(Zi_high), levels=levels, cmap='plasma')
    axs[0].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[0]), transform=axs[0].transAxes, fontsize=16)
    if mission == 'GF':
        axs[0].text(xi.mean(), yi.mean(), 'NO DATA', ha='center', va='center', color='k', fontsize=16)
    else:
        pass

    medium = axs[1].contourf(xi, yi, np.log10(Zi_medium), levels=levels, cmap='plasma')
    axs[1].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[1]), transform=axs[1].transAxes, fontsize=16)

    low = axs[2].contourf(xi, yi, np.log10(Zi_low), levels=levels, cmap='plasma')
    axs[2].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[2]), transform=axs[2].transAxes, fontsize=16)

    cbar = plt.colorbar(high, ax=axs)
    cbar.set_label('Log Ne [$m^{-3}$]', fontsize=16)

    plt.yticks(np.arange(-60, 61, 30))
    plt.xticks(np.arange(0, 25, 2))

    # plt.xlabel('Magnetic Local time [hours]', fontsize=16)
    # axs[0].set_ylabel('Latitude [degrees]', fontsize=16)
    # axs[1].set_ylabel('Latitude [degrees]', fontsize=16)
    # axs[2].set_ylabel('Latitude [degrees]', fontsize=16)

    # st = fig.suptitle(f'{mission_full_name}'.format(mission_full_name=mission_full_name), fontsize=16)
    # st.set_y(0.96)

    # plt.show()
    fig.supylabel('Magnetic Latitude [degrees]', fontsize=16,x=0.05)
    fig.supxlabel('Magnetic Local time [hours]', fontsize=16)
    # fig.tight_layout()

    plt.savefig('../figures/v2/heatmap_{mission}.png'.format(mission=mission))
    plt.close()


plot_heatmap('GR')
plot_heatmap('GF')


def plot_semiorbits_mean(mission):

    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass


    # fig, axes = plt.subplots(8,3, figsize=(20, 16), sharex=True, sharey=True)
    # axs = axes.flatten()
    fig, axes = plt.subplots(4,1, figsize=(12, 8), sharex=True, sharey=True)
    axs = axes.flatten()
    mlat = np.arange(-90, 90, 2.5)

    cmap = plt.get_cmap('plasma')
    slicedCM = cmap(np.linspace(0, 1, 24))

    for id_ax in range(4):

        if id_ax == 0:
            loop_time = np.arange(0,6)
        if id_ax == 1:
            loop_time = np.arange(6, 12)
        if id_ax == 2:
            loop_time = np.arange(12,18)
        if id_ax == 3:
            loop_time = np.arange(18, 24)


        for hour in range(0,24):

            df = pd.read_csv('../tables/v2/{mission}_semiorbits_{hour}_v2.csv'.format(mission=mission, hour=hour))

            df['Absolute_Nel'] = np.log10(df['Absolute_Ne'])

            absolute_ne = (df.groupby(pd.cut(df['mlat'], bins=np.arange(-90, 91,2.5)))['Absolute_Nel'].mean()).values
            axs[id_ax].plot(mlat, absolute_ne, color='gray', ls ='dashed', alpha = 0.5)

        for hour in loop_time:

            df = pd.read_csv('../tables/v2/{mission}_semiorbits_{hour}_v2.csv'.format(mission=mission, hour=hour))

            df['Absolute_Nel'] = np.log10(df['Absolute_Ne'])

            absolute_ne = (df.groupby(pd.cut(df['mlat'], bins=np.arange(-90, 91,2.5)))['Absolute_Nel'].mean()).values
            axs[id_ax].plot(mlat, absolute_ne, color= slicedCM[hour], label = ('Hour {hour}').format(hour=hour), lw=2)

        axs[id_ax].legend(loc='upper right')
        axs[id_ax].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[id_ax]),
                    transform=axs[id_ax].transAxes, fontsize=16)


    plt.xlim(-90, 90)
    plt.xticks(np.arange(-90, 91, 15))

    plt.ylim(10.25,12)
    # plt.yticks(np.arange(10.25,11.51,0.25))


    fig.supylabel('Log Ne [$m^{-3}$]', fontsize=16) #,x=0.05)
    fig.supxlabel('Magnetic Latitude [degrees]', fontsize=16)

    fig.tight_layout()
    # plt.show()
    plt.savefig('../figures/v2/{mission}_maglat_Nel_fig_all_mean.png'.format(mission=mission))
    plt.close()

# plot_semiorbits_mean('GF')
# plot_semiorbits_mean('GR')


