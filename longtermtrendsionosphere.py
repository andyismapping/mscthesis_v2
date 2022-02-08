import glob
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from astropy.convolution import Gaussian2DKernel, convolve
import matplotlib as mpl
import os


import warnings
import string

warnings.filterwarnings(action='once')


def process_heatmap(mission, method, level):
    f107 = pd.read_csv('../data/external/f107/Kp_ap_Ap_SN_F107_since_1932.txt', delimiter=r"\s+", comment='#')
    f107['date_simplified'] = f107.apply(lambda x: datetime.datetime(int(x['YYYY']), int(x['MM']), int(x['DD'])),
                                         axis=1)
    if level == 'low':
        f107 = f107[(f107['F10.7obs'] < 80)]
    elif level == 'medium':
        f107 = f107[(f107['F10.7obs'] >= 80) & (f107['F10.7obs'] < 120)]
    elif level == 'high':
        f107 = f107[(f107['F10.7obs'] >= 120)]
    else:
        print('wrong level')

    if mission == 'GR':
        f107 = f107[(f107['YYYY'] >= 2002) & (f107['YYYY'] <= 2017)]
    elif mission == 'GF':
        f107 = f107[(f107['YYYY'] >= 2018) & (f107['YYYY'] <= 2020)]
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
            df = pd.read_csv(df_file[0], usecols=['Latitude', 'mlt', 'Absolute_Ne'])
            df_plot = df_plot.append(df)

        except:
            pass

    yi = np.arange(-60, 60, 0.1)
    xi = np.arange(0, 24, 0.1)
    xi, yi = np.meshgrid(xi, yi)

    df_plot = df_plot.dropna()

    zi = griddata((df_plot['mlt'], df_plot['Latitude']), df_plot['Absolute_Ne'], (xi, yi), method=method)
    np.save("../tables/{mission}_{level}_{method}.npy".format(mission=mission, level=level, method=method), zi)


# process_heatmap(mission='GR', method='nearest', level='high')
# process_heatmap(mission='GR', method='nearest', level='medium')
# process_heatmap(mission='GR', method='nearest', level='low')
# process_heatmap(mission='GF', method='nearest', level='high')
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

    g = Gaussian2DKernel(5, 5)

    try:
        Z_low = np.load('../tables/{mission}_low_nearest.npy'.format(mission=mission))
        Zi_low = convolve(Z_low, g)
    except:
        print('Z_low not found')
        Z_low = np.full(xi.shape, np.nan)
        Zi_low = np.full(xi.shape, np.nan)

    try:
        Z_medium = np.load('../tables/{mission}_medium_nearest.npy'.format(mission=mission))
        Zi_medium = convolve(Z_medium, g)
    except:
        print('Z_medium not found')
        Z_medium = np.full(xi.shape, np.nan)
        Zi_medium = np.full(xi.shape, np.nan)

    try:
        Z_high = np.load('../tables/{mission}_high_nearest.npy'.format(mission=mission))
        Zi_high = convolve(Z_high, g)
    except:
        print('Z_high not found')
        Z_high = np.full(xi.shape, np.nan)
        Zi_high = np.full(xi.shape, np.nan)
        pass

    levels = np.arange(10, 12.5, 0.25)


    fig, axs = plt.subplots(3, figsize=(12, 8), sharex=True, sharey=True)



    high = axs[0].contourf(xi, yi, np.log10(Zi_high), levels=levels, cmap='plasma')
    axs[0].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[0]), transform=axs[0].transAxes, fontsize=12)

    medium = axs[1].contourf(xi, yi, np.log10(Zi_medium), levels=levels, cmap='plasma')
    axs[1].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[1]), transform=axs[1].transAxes, fontsize=12)

    low = axs[2].contourf(xi, yi, np.log10(Zi_low), levels=levels, cmap='plasma')
    axs[2].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[2]), transform=axs[2].transAxes, fontsize=12)

    cbar = plt.colorbar(high, ax=axs)
    cbar.set_label('Log Ne [$m^{-3}$]', fontsize=12)

    plt.yticks(np.arange(-60, 61, 30))
    plt.xticks(np.arange(0, 25, 2))

    plt.xlabel('Magnetic Local time [hours]', fontsize=12)
    axs[0].set_ylabel('Latitude [degrees]', fontsize=12)
    axs[1].set_ylabel('Latitude [degrees]', fontsize=12)
    axs[2].set_ylabel('Latitude [degrees]', fontsize=12)

    # st = fig.suptitle(f'{mission_full_name}'.format(mission_full_name=mission_full_name), fontsize=16)
    # st.set_y(0.96)

    fig.tight_layout
    # plt.show()
    plt.savefig('../figures/v2/heatmap_{mission}.png'.format(mission=mission))
    plt.close()


# plot_heatmap('GR')
# plot_heatmap('GF')

def process_semiorbits(mission):

    df_list = glob.glob(
                '../data/processed/Absolute_Ne_v2/*/{mission}_OPER_NE__KBR_2F_*.cdf.csv'.format(mission=mission))

    for df_file in df_list:
        df = pd.read_csv(df_file, usecols=['Timestamp','mlat', 'mlt', 'Absolute_Ne'])

        df = df.dropna()

        df['hour']= df.mlt.apply(lambda x : int(x))

        for hour in range(0,24):
            df_hour = df[df['hour'] == hour]

            output = f'../tables/v2/{mission}_semiorbits_{hour}_v2.csv'

            # output = '../tables/semiorbits_{hour}.csv'.format(hour=hour)

            df_hour.to_csv(output, mode='a', header=not os.path.exists(output), index=False)


process_semiorbits('GR')
process_semiorbits('GF')

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def plot_semiorbits(mission, nfig):

    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    hours = np.arange(0,24,1)
    hours_idx= np.array_split(hours, nfig)

    for i in range(0,nfig):
        list_hours = hours_idx[i]
        nrows = len(list_hours)

        fig, axes = plt.subplots(nrows, 3, figsize=(16, 2*nrows), sharex=True)
        axs = axes.flatten()

        id_ax = 0

        for hour in list_hours:

            df = pd.read_csv('../tables/{mission}_semiorbits_{hour}_v2.csv'.format(mission=mission, hour=hour))

            # bin lat
            df['mlat'] = np.round(df['mlat'], 0)

            # extract mean, max, min
            df_groupby = df.groupby('mlat').agg({'Absolute_Ne': ['mean', 'min', 'max']})

            # ax1 = plt.subplot(1, 3, 1)
            axs[id_ax].set_xlim(-90, 90)
            axs[id_ax].scatter(df['mlat'], df['Absolute_Ne'],s=10)
            # ax1.set_ylim(-10e11, 10e11)
            # axs[id_ax].set_xlabel("Magnetic Latitude[degrees]", fontsize=12)
            # axs[id_ax].set_ylabel("Ne [$m^{-3}$]", fontsize=12)

            id_ax += 1

            # ax2 = plt.subplot(1, 3, 2)
            axs[id_ax].set_xlim(-90, 90)
            axs[id_ax].plot(df_groupby.index, df_groupby['Absolute_Ne']['mean'])
            # ax2.set_ylim(-10e10, 10e10)
            axs[id_ax].set_title('Hour {hour}'.format(hour=hour), fontsize=12)
            # axs[id_ax].set_xlabel("Magnetic Latitude[degrees]", fontsize=12)
            # axs[id_ax].set_ylabel("Ne [$m^{-3}$]", fontsize=12)

            id_ax += 1

            # ax3 = plt.subplot(1, 3, 3)
            axs[id_ax].set_xlim(-90, 90)
            axs[id_ax].plot(df_groupby.index, df_groupby['Absolute_Ne']['mean'])
            axs[id_ax].plot(df_groupby.index, df_groupby['Absolute_Ne']['max'])
            axs[id_ax].plot(df_groupby.index, df_groupby['Absolute_Ne']['min'])
            # ax3.set_ylim(-10e11, 10e11)
            # axs[id_ax].set_xlabel("Magnetic Latitude[degrees]", fontsize=12)
            # axs[id_ax].set_ylabel("Ne [$m^{-3}$]", fontsize=12)

            id_ax += 1


        plt.suptitle(f'{mission_full_name}', fontsize=16,fontweight="bold")
        # set labels
        mpl.rcParams.update({'font.size': 12})
        plt.setp(axes[-1, :], xlabel="Magnetic Latitude[degrees]")
        plt.setp(axes[:, 0], ylabel="Ne [$m^{-3}$]")
        fig.tight_layout()
        # plt.show()
        plt.savefig('../figures/{mission}_maglat_Ne_fig_{fig}.png'.format(mission=mission, fig=i))
        plt.close()

# plot_semiorbits('GR',nfig=4)
# plot_semiorbits('GF',nfig=4)

