import string

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl
import datetime
import math
import aacgmv2
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import kde
import glob


def plot_hist(mission):
    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()

    df = df.sort_values('Location')

    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)
    axs = axs.flatten()

    # all radars
    axs[0].hist(df['nel_diff'], bins=np.arange(-2, 2.5, 0.1),density=False)
    axs[0].set_title('All Radars', fontsize=14)
    axs[0].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[0]),
                      transform=axs[0].transAxes, fontsize=12)
    # density = kde.gaussian_kde(df['nel_diff'])
    # x = np.arange(-2, 2.5, 0.1)
    # y = density(x)
    # axs[0].plot(x,y)

    for i_radar, radar_name in enumerate(df.Location.unique()):
        j_radar = i_radar+1
        df_plot = df[df.Location == radar_name]
        axs[j_radar].hist(df_plot['nel_diff'], bins = np.arange(-2, 2.5 , 0.1),density=False)
        axs[j_radar].set_title('{radar}'.format(radar=radar_name), fontsize=14)
        axs[j_radar].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[j_radar]), transform=axs[j_radar].transAxes, fontsize=14)


        # density = kde.gaussian_kde(df_plot['nel_diff'])
        # x = np.arange(-2, 2.5, 0.1)
        # y = density(x)
        # axs[j_radar].plot(x, y)

    fig.supylabel('Count', fontsize=14)
    if mission == 'GR':
        fig.supxlabel('$Ne_{GR}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)
    elif mission == 'GF':
        fig.supxlabel('$Ne_{GF}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)

    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/v2/hist_{mission}_individual.png".format(mission=mission))
    plt.close()

# plot_hist('GF')
# plot_hist('GR')

def plot_hist_single():



    # fig, axs = plt.subplots()
    # axs = axs.flatten()

    for id_ax in range(0,2):

        if id_ax == 0:
            df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission='GR')).drop(
                ['Unnamed: 0'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
            fig, axs = plt.subplots()
            axs.set_xlabel('$Ne_{GR}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)
        else:
            df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission='GF')).drop(
                ['Unnamed: 0'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
            fig, axs = plt.subplots()
            axs.set_xlabel('$Ne_{GF}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)

        # all radars
        axs.hist(df['nel_diff'], bins=np.arange(-2, 2.5, 0.1),density=False)
        # axs.text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[id_ax]),
        #                   transform=axs[id_ax].transAxes, fontsize=14)
        axs.set_xticks(np.arange(-2, 2.5, 0.5))
        axs.set_ylabel('Count', fontsize=14)





        # fig.supylabel('Count', fontsize=14)
        # fig.supxlabel('$Ne_{GR}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)
        # if mission == 'GR':
        #     fig.supxlabel('$Ne_{GR}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)
        # elif mission == 'GF':
        #     plt.xlabel('$Ne_{GF}$ - Log $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)


        fig.tight_layout()
        # plt.show()
        plt.savefig("../figures/v2/hist_single_{idx}.png".format(idx=id_ax))
        plt.close()

# plot_hist_single()

def plot_density(mission):
    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()

    df = df.sort_values('Location')

    fig, axs = plt.subplots()

    for i_radar, radar_name in enumerate(df.Location.unique()):
        df_plot = df[df.Location == radar_name]

        density = kde.gaussian_kde(df_plot['nel_diff'])
        x = np.arange(-2, 2.5, 0.1)
        y = density(x)
        axs.plot(x, y, label = radar_name)


    fig.supylabel('Density', fontsize=14)
    if mission == 'GR':
        fig.supxlabel('Log $Ne_{GR}$ - $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)
    elif mission == 'GF':
        fig.supxlabel('Log $Ne_{GF}$ - $Ne_{RADAR}$ [$m^{-3}$]', fontsize=14)

    density = kde.gaussian_kde(df['nel_diff'])
    x = np.arange(-2, 2.5, 0.1)
    y = density(x)
    axs.plot(x,y,'k', ls='dashed', lw=3, label = 'All Radars')

    plt.legend()
    axs.set_xticks(np.arange(-2, 2.5, 0.5))

    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/v2/density_{mission}_individual.png".format(mission=mission))
    plt.close()

# plot_density('GF')
# plot_density('GR')



# estimate local time zone
def calc_localtime(longitude_local, datetime_UTC):
    longitudes = []
    for i in iter(range(0, 13)):
        longitudes.append(7.5 + (15 * i))

    longitudes.append(np.abs(longitude_local))
    longitudes.sort()

    idx_localtime = np.argwhere(longitudes == np.abs(longitude_local))[0][0]
    if longitude_local >= 0:
        datetime_local = datetime_UTC + datetime.timedelta(hours=int(idx_localtime))
    else:
        datetime_local = datetime_UTC - datetime.timedelta(hours=int(idx_localtime))

    return datetime_local

def plot_timeday(mission):

    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'],
                                                                                                   axis=1)
    df['date'] = pd.to_datetime(df['date'])

    df['date_local'] = df.apply(lambda x: calc_localtime(x.Longitude, x.date), axis=1)

    df['hour'] = df.date.apply(lambda x: (x.hour + x.minute/60 + x.second/3600))

    # one big graph
    fig, ax = plt.subplots(figsize=(12, 5))

    for location in np.unique(df['Location']):
        df_plot = df[df['Location'] == location]
        plt.scatter(df_plot['hour'], df_plot['nel_diff'], label=location)

    plt.xlim(0,24)
    plt.xticks(np.arange(0,24))
    plt.ylim(-3,3)
    plt.legend()

    plt.xlabel('Local time [hours]', fontsize=14)
    if mission == 'GR':
        plt.ylabel("Difference $Ne_{GR}$ - $Ne_{RADAR}$ [$m^{-3}$]", fontsize=14)
    elif mission == 'GF':
        plt.ylabel("Difference $Ne_{GF}$ - $Ne_{RADAR}$ [$m^{-3}$]", fontsize=14)

    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/v2/timeday_{mission}_all.png".format(mission=mission))
    plt.close()

    fig, ax = plt.subplots()
    plt.hist(df['hour'], bins = np.arange(0,24))
    plt.savefig("../figures/v2/hist_timeday_{mission}_all.png".format(mission=mission))
    plt.close()

# plot_timeday('GR')
# plot_timeday('GF')

def plot_density_timeday(mission):
    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()

    df['date_local'] = df.apply(lambda x: calc_localtime(x.Longitude, x.date), axis=1)

    df['hour'] = df.date.apply(lambda x: (x.hour + x.minute/60 + x.second/3600))

    df = df.sort_values('Location')

    fig, axs = plt.subplots()

    for i_radar, radar_name in enumerate(df.Location.unique()):
        df_plot = df[df.Location == radar_name]

        density = kde.gaussian_kde(df_plot['hour'])
        x = np.arange(0,24,0.1)
        y = density(x)
        axs.plot(x, y, label = radar_name)


    fig.supylabel('Density', fontsize=14)
    fig.supxlabel('Local time [hours]')
    # if mission == 'GR':
    #     fig.supxlabel('Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]', fontsize=14)
    # elif mission == 'GF':
    #     fig.supxlabel('Log $Ne_{RADAR}$ - $Ne_{GF}$ [$m^{-3}$]', fontsize=14)

    density = kde.gaussian_kde(df['hour'])
    x = np.arange(0,24, 0.1)
    y = density(x)
    axs.plot(x,y,'k', ls='dashed', lw=3, label = 'All Radars')
    plt.xlim(0,24)
    plt.xticks(np.arange(0,25,2))

    plt.legend()

    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/v2/densitytimeday_{mission}_individual.png".format(mission=mission))
    plt.close()

# plot_density_timeday('GF')
# plot_density_timeday('GR')


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

    mlat, mlon, mlt = aacgmv2.wrapper.get_aacgm_coord(latitude, longitude, altitude, timestamp, method='’ALLOWTRACE|GEOCENTRIC’')

    # mlat, mlon, _ = aacgmv2.wrapper.convert_latlon(latitude, longitude, altitude, timestamp, method_code="G2A|ALLOWTRACE")
    # mlt = np.nan if np.isnan(mlon) else aacgmv2.wrapper.convert_mlt(mlon, timestamp, m2a=False)[0]

    return mlat, mlon, mlt


def process_magtimeday(mission):
    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/{mission}_conjunctions_clean.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)

    df["radar_mnemonic_unique"] = df["radar_mnemonic"].str[0:3]

    df['location'] = df['radar_mnemonic_unique']
    df['longitude'] = df['radar_mnemonic_unique']
    df['latitude'] = df['radar_mnemonic_unique']

    for i in iter(range(0, len(df))):

        radar = df['radar_mnemonic_unique'][i]

        if str(radar)[0:3] == 'mlh':
            df['location'][i] = 'Millstone Hill'
            df['longitude'][i] = 288.51
            df['latitude'][i] = 42.619
        elif str(radar[0:3]) == 'arg':
            df['location'][i] = 'Arecibo'
            df['longitude'][i] = 293.25
            df['latitude'][i] = 18.345
        elif str(radar[0:3]) == 'arl':
            df['location'][i] = 'Arecibo'
            df['longitude'][i] = 293.25
            df['latitude'][i] = 18.345
        elif str(radar[0:3]) == 'eis':
            df['location'][i] = 'Tronsø'
            df['longitude'][i] = 19.21
            df['latitude'][i] = 69.6
        elif str(radar[0:3]) == 'tro':
            df['location'][i] = 'Tronsø'
            df['longitude'][i] = 19.21
            df['latitude'][i] = 69.6
        elif str(radar[0:3]) == 'jro':
            df['location'][i] = 'Jicamarca'
            df['longitude'][i] = 283.13
            df['latitude'][i] = -11.95
        elif str(radar[0:3]) == 'kpi':
            df['location'][i] = 'Kharkov'
            df['longitude'][i] = 36.2
            df['latitude'][i] = 50.0
        elif str(radar[0:3]) == 'mui':
            df['location'][i] = 'MU'
            df['longitude'][i] = 136.1
            df['latitude'][i] = 34.8
        elif str(radar[0:3]) == 'ran':
            df['location'][i] = 'Resolute Bay'
            df['longitude'][i] = 265.09424
            df['latitude'][i] = 74.72955
        elif str(radar[0:3]) == 'ras':
            df['location'][i] = 'Resolute Bay'
            df['longitude'][i] = 265.09424
            df['latitude'][i] = 74.72955
        elif str(radar[0:3]) == 'lyr':
            df['location'][i] = 'Svalbard'
            df['longitude'][i] = 16.02
            df['latitude'][i] = 78.09
        elif str(radar[0:3]) == 'pfa':
            df['location'][i] = 'Poker Flat'
            df['longitude'][i] = 212.529
            df['latitude'][i] = 65.13

    df = df[~df.location.str.contains("Kharkov", na=False)]
    df = df[~df.location.str.contains("MU", na=False)]

    df['date'] = pd.to_datetime(df['date'])

    df['longitude'] = df['longitude'].astype(float)

    df['Re'] = df.latitude.apply(lambda x: earth_radius(x))

    df[['mlat', 'mlon', 'mlt']] = (df.apply(
        lambda x: geo2mag(x['latitude'], x['longitude'], 500, x['date']),
        axis=1).values.tolist())

    df.to_csv('../tables/{mission}_conjunctions_clean_magcoords.csv'.format(mission=mission))

# process_magtimeday('GR')
# process_magtimeday('GF')

def plot_magtimeday(mission):

    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    df = pd.read_csv('../tables/{mission}_conjunctions_clean_magcoords.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)

    # one big graph
    fig, ax = plt.subplots(figsize=(12, 5))

    for location in np.unique(df['location']):
        df_plot = df[df['location'] == location]
        plt.scatter(df_plot['mlt'], df_plot['ne_diff'], label=location)

    plt.xlim(0,24)
    plt.xticks(np.arange(0,24))
    plt.legend()

    plt.title('{mission_full_name} Conjunctions'.format(mission_full_name=mission_full_name), fontsize=16)
    plt.xlabel('Magnetic local time [hours]', fontsize=12)
    if mission == 'GR':
        plt.ylabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    elif mission == 'GF':
        plt.ylabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)

    # plt.show()
    plt.savefig("../figures/magtimeday_{mission}_all.png".format(mission=mission))
    plt.close()

# plot_magtimeday('GR')
# plot_magtimeday('GF')

def plot_timeseries(mission):
    mpl.rcParams.update({'font.size': 16})

    if mission == 'GR':
        mission_full_name = 'GRACE'
        gf_files = glob.glob('../data/processed/Absolute_Ne_v2/GRACE/*.csv')
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
        gf_files = glob.glob('../data/processed/Absolute_Ne_v2/GRACEFO/*.csv')
    else:
        print('Mission not recognized')
        pass

    # one big graph
    fig, ax = plt.subplots(figsize=(20, 8))

    # for gf_file in gf_files:
    for i in range(0,len(gf_files)):
        gf_file = gf_files[i]
        if i == 0:

            gf = pd.read_csv(gf_file)
            gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])
            gf['Absolute_Nel'] = np.log10(gf['Absolute_Ne'])

            plt.scatter(gf['Timestamp'], gf['Absolute_Nel'], color='k',s=0.5, label = mission)
        else:
            gf = pd.read_csv(gf_file)
            gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])
            gf['Absolute_Nel'] = np.log10(gf['Absolute_Ne'])

            plt.scatter(gf['Timestamp'], gf['Absolute_Nel'], color='k', s=0.5)


    df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)
    df['date'] = pd.to_datetime(df['date'])


    for location in np.unique(df['Location']):
        df_plot = df[df['Location'] == location]
        plt.scatter(df_plot['date'], df_plot['nel_radar'], label=location)

    ax.xaxis_date()

    # months = mdates.MonthLocator(interval=12)
    # ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    # ax.set_xlim(pd.Timestamp("2002-01"), pd.Timestamp("2020-12"))

    plt.legend(loc='lower left')

    # plt.title('GRACE and GRACE-FO Conjunctions', fontsize=16)
    plt.xlabel('Dates', fontsize=18)
    plt.ylabel("Log $Ne$ [$m^{-3}$]", fontsize=18)

    plt.tight_layout()

    # plt.show()
    plt.savefig("../figures/v2/timeseries_{mission}.png".format(mission=mission))
    plt.close()

# plot_timeseries('GR')
# plot_timeseries('GF')

def plot_density_timeday_hour(mission):
    cmap = plt.get_cmap('GnBu')
    slicedCM = cmap(np.linspace(0, 1, 24))

    if mission == 'GR':
        mission_full_name = 'GRACE'
    elif mission == 'GF':
        mission_full_name = 'GRACE-FO'
    else:
        print('Mission not recognized')
        pass

    fig, axs = plt.subplots(1, 2,figsize=(12, 5))
    axs = axs.flatten()

    for id_ax in range(0,2):

        if id_ax == 1:

            df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()

            df['date_local'] = df.apply(lambda x: calc_localtime(x.Longitude, x.date), axis=1)

            df['hour'] = df.date.apply(lambda x: (x.hour + x.minute/60 + x.second/3600))

            df['hour'] = df.hour.apply(lambda x: int(x))

            df = df.sort_values('hour')

            # fig, axs = plt.subplots()

            for i_radar, radar_hour in enumerate(df.hour.unique()):
                df_plot = df[df.hour == radar_hour]

                try:
                    density = kde.gaussian_kde(df_plot['nel_diff'])
                    x = np.arange(-2, 2.5, 0.1)
                    y = density(x)
                    axs[id_ax].plot(x, y, label = radar_hour, color = slicedCM[radar_hour])

                except:
                    pass

            density = kde.gaussian_kde(df['nel_diff'])
            x = np.arange(-2, 2.5, 0.1)
            y = density(x)
            axs[id_ax].plot(x, y, 'k', ls='dashed', lw=3, label='All hours')

            axs[id_ax].legend(ncol=2)
            axs[id_ax].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[id_ax]),
                            transform=axs[id_ax].transAxes, fontsize=14)

            axs[id_ax].set_xticks(np.arange(-2, 2.5, 0.5))

            axs[id_ax].set_ylabel('Density')

            if mission == 'GR':
                axs[id_ax].set_xlabel('Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]', fontsize=14)
            elif mission == 'GF':
                axs[id_ax].set_xlabel('Log $Ne_{RADAR}$ - $Ne_{GF}$ [$m^{-3}$]', fontsize=14)

        elif id_ax == 0:

            df = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission=mission)).drop(
                ['Unnamed: 0'], axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()

            df['date_local'] = df.apply(lambda x: calc_localtime(x.Longitude, x.date), axis=1)

            df['hour'] = df.date.apply(lambda x: (x.hour + x.minute / 60 + x.second / 3600))

            df = df.sort_values('Location')

            # fig, axs = plt.subplots()

            for i_radar, radar_name in enumerate(df.Location.unique()):
                df_plot = df[df.Location == radar_name]

                density = kde.gaussian_kde(df_plot['hour'])
                x = np.arange(0, 24, 0.1)
                y = density(x)
                axs[id_ax].plot(x, y, label=radar_name)


            density = kde.gaussian_kde(df['hour'])
            x = np.arange(0, 24, 0.1)
            y = density(x)
            axs[id_ax].plot(x, y, 'k', ls='dashed', lw=3, label='All Radars')
            axs[id_ax].set_xlim(0, 24)
            axs[id_ax].set_xticks(np.arange(0, 25, 2))

            axs[id_ax].legend()

            axs[id_ax].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[id_ax]),
                        transform=axs[id_ax].transAxes, fontsize=14)

            axs[id_ax].set_ylabel('Density', fontsize=14)
            axs[id_ax].set_xlabel('Local time [hours]', fontsize=14)



    # fig.supylabel('Density', fontsize=14)
    # # fig.supxlabel('Local time [hours]')
    # if mission == 'GR':
    #     fig.supxlabel('Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]', fontsize=14)
    # elif mission == 'GF':
    #     fig.supxlabel('Log $Ne_{RADAR}$ - $Ne_{GF}$ [$m^{-3}$]', fontsize=14)


    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/v2/densitytimeday_{mission}_double.png".format(mission=mission))
    plt.close()

# plot_density_timeday_hour('GF')
# plot_density_timeday_hour('GR')

def plot_timeseries_together():
    mpl.rcParams.update({'font.size': 16})

    # if mission == 'GR':
    #     mission_full_name = 'GRACE'
    #     gf_files = glob.glob('../data/processed/Absolute_Ne_v2/GRACE/*.csv')
    # elif mission == 'GF':
    #     mission_full_name = 'GRACE-FO'
    #     gf_files = glob.glob('../data/processed/Absolute_Ne_v2/GRACEFO/*.csv')
    # else:
    #     print('Mission not recognized')
    #     pass

    gf_files = glob.glob('../data/processed/Absolute_Ne_v2/*/*.csv')

    # one big graph
    fig, ax = plt.subplots(figsize=(20, 8))

    # for gf_file in gf_files:
    for i in range(0,len(gf_files)):
        gf_file = gf_files[i]
        if i == 0:

            gf = pd.read_csv(gf_file)
            gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])
            gf['Absolute_Nel'] = np.log10(gf['Absolute_Ne'])

            plt.scatter(gf['Timestamp'], gf['Absolute_Nel'], color='k',s=0.5)
        else:
            gf = pd.read_csv(gf_file)
            gf['Timestamp'] = pd.to_datetime(gf['Timestamp'])
            gf['Absolute_Nel'] = np.log10(gf['Absolute_Ne'])

            plt.scatter(gf['Timestamp'], gf['Absolute_Nel'], color='k', s=0.5)


    df_gr = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission='GR')).drop(['Unnamed: 0'], axis=1)
    df_gf = pd.read_csv('../tables/v2/conjunctions_clean_{mission}.csv'.format(mission='GF')).drop(['Unnamed: 0'], axis=1)
    df = pd.concat([df_gr, df_gf], ignore_index=True, sort=True)

    df['date'] = pd.to_datetime(df['date'])


    for location in np.unique(df['Location']):
        df_plot = df[df['Location'] == location]
        plt.scatter(df_plot['date'], df_plot['nel_radar'], label=location)

    plt.ylim(4,17)

    ax.hlines(y=14, xmin=datetime.datetime(2002, 4, 4), xmax=datetime.datetime(2017, 6, 29), linewidth=4,
              color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
    ax.hlines(y=14, xmin=datetime.datetime(2018, 5, 29), xmax=datetime.datetime(2020, 12, 27), linewidth=4,
              color=plt.rcParams["axes.prop_cycle"].by_key()["color"][2])

    plt.text(datetime.datetime(2009, 12, 1), 15, 'GRACE', ha='center', va='center',
             color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0], fontsize=16)
    plt.text(datetime.datetime(2019, 9, 1), 15, 'GRACE-FO', ha='center', va='center',
             color=plt.rcParams["axes.prop_cycle"].by_key()["color"][2], fontsize=16)

    ax.xaxis_date()

    # months = mdates.MonthLocator(interval=12)
    # ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    # ax.set_xlim(pd.Timestamp("2002-01"), pd.Timestamp("2020-12"))

    plt.legend(loc='lower right')

    # plt.title('GRACE and GRACE-FO Conjunctions', fontsize=16)
    plt.xlabel('Dates', fontsize=18)
    plt.ylabel("Log $Ne$ [$m^{-3}$]", fontsize=18)

    plt.tight_layout()

    # plt.show()
    plt.savefig("../figures/v2/timeseries.png")
    plt.close()

plot_timeseries_together()