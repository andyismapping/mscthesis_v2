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

    df = df.sort_values('Location')

    fig, axs = plt.subplots(4, 2, figsize=(9, 12))
    axs = axs.flatten()

    # all radars
    axs[0].hist(df['nel_diff'], bins=np.arange(-2, 2.5, 0.5))
    axs[0].set_title('all RADARS', fontsize=14)
    axs[0].set_xticks(np.arange(-2, 2.5, 0.5))
    ya = axs[0].get_yaxis()
    ya.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel('Count', fontsize=12)
    # axs[i_radar].set_xlabel(' Ne [$m^{-3}$]', fontsize=12)
    if mission == 'GR':
        axs[0].set_xlabel("Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    elif mission == 'GF':
        axs[0].set_xlabel("Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)

    axs[0].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[0]),
                      transform=axs[0].transAxes, fontsize=12)

    for i_radar, radar_name in enumerate(df.Location.unique()):
        j_radar = i_radar+1
        df_plot = df[df.Location == radar_name]
        # axs[i_radar].hist(df_plot['ne_diff'],
        #                   bins=[-4.5e11, -3.5e11, -2.5e11, -1.5e11, -0.5e11, 0.5e11, 1.5e11, 2.5e11, 3.5e11, 4.5e11])
        axs[j_radar].hist(df_plot['nel_diff'], bins = np.arange(-2, 2.5 , 0.5))
        axs[j_radar].set_title('{radar}'.format(radar=radar_name), fontsize=14)
        axs[j_radar].set_xticks(np.arange(-2, 2.5 , 0.5))
        ya = axs[j_radar].get_yaxis()
        ya.set_major_locator(MaxNLocator(integer=True))
        axs[j_radar].set_ylabel('Count', fontsize=12)
        # axs[i_radar].set_xlabel(' Ne [$m^{-3}$]', fontsize=12)
        if mission == 'GR':
            axs[j_radar].set_xlabel("Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
        elif mission == 'GF':
            axs[j_radar].set_xlabel("Log $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)

        axs[j_radar].text(0.05, 1.05, '{letter})'.format(letter=string.ascii_lowercase[j_radar]), transform=axs[j_radar].transAxes, fontsize=12)


    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/v2/hist_{mission}_individual.png".format(mission=mission))
    plt.close()

    # plot separate graphs
    # fig, ax = plt.subplots(figsize=(7.5, 5))
    # plt.title('{radar}'.format(radar=radar_name))
    #
    # df_plot = df[df.location == radar_name]
    #
    # plt.hist(df_plot['ne_diff'],
    #          bins=[-4.5e11, -3.5e11, -2.5e11, -1.5e11, -0.5e11, 0.5e11, 1.5e11, 2.5e11, 3.5e11, 4.5e11])
    # plt.xticks(np.arange(-4e11, 5e11, 1e11))
    # ya = ax.get_yaxis()
    # ya.set_major_locator(MaxNLocator(integer=True))
    # plt.ylabel('Count')
    # plt.xlabel('Ne [$m^{-3}$]')
    #
    # # plt.show()
    # plt.savefig("../figures/diff_{mission}_{name}.png".format(mission=mission, name=radar_name))
    # plt.close()
    # pass

    # fig, ax = plt.subplots(figsize=(7.5, 5))
    # plt.title('{mission_full_name} - all RADARS'.format(mission_full_name=mission_full_name), fontsize=16)
    # plt.hist(df['nel_diff'])
    # # plt.hist(df['ne_diff'], bins=[-4.5e11, -3.5e11, -2.5e11, -1.5e11, -0.5e11, 0.5e11, 1.5e11, 2.5e11, 3.5e11, 4.5e11])
    # # plt.xticks(np.arange(-4e11, 5e11, 1e11))
    # plt.ylabel('Count', fontsize=12)
    # if mission == 'GR':
    #     plt.xlabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    # elif mission == 'GF':
    #     plt.xlabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    #
    # # $Ne_{RADAR}$
    # fig.tight_layout()
    # # plt.show()
    # plt.savefig("../figures/v2/hist_{mission}_sum.png".format(mission=mission))
    # plt.close()


plot_hist('GF')
plot_hist('GR')

def plot_scatter(mission):
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

    for i in iter(range(0, len(df))):

        radar = df['radar_mnemonic_unique'][i]

        if str(radar)[0:3] == 'mlh':
            df['location'][i] = 'Millstone Hill'
        elif str(radar[0:3]) == 'arg':
            df['location'][i] = 'Arecibo'
        elif str(radar[0:3]) == 'arl':
            df['location'][i] = 'Arecibo'
        elif str(radar[0:3]) == 'eis':
            df['location'][i] = 'Tronsø'
        elif str(radar[0:3]) == 'tro':
            df['location'][i] = 'Tronsø'
        elif str(radar[0:3]) == 'jro':
            df['location'][i] = 'Jicamarca'
        elif str(radar[0:3]) == 'kpi':
            df['location'][i] = 'Kharkov'
        elif str(radar[0:3]) == 'mui':
            df['location'][i] = 'MU'
        elif str(radar[0:3]) == 'ran':
            df['location'][i] = 'Resolute Bay'
        elif str(radar[0:3]) == 'ras':
            df['location'][i] = 'Resolute Bay'
        elif str(radar[0:3]) == 'lyr':
            df['location'][i] = 'Svalbard'
        elif str(radar[0:3]) == 'pfa':
            df['location'][i] = 'Poker Flat'

    df = df[~df.location.str.contains("Kharkov", na=False)]
    df = df[~df.location.str.contains("MU", na=False)]

    # small individual graphs in one big figure
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharex=True, sharey=True)
    axs = axes.flatten()
    # histogram
    plt.suptitle('{mission_full_name} Conjunctions'.format(mission_full_name=mission_full_name), fontsize=16)

    for i_radar, radar_name in enumerate(df.location.unique()):
        df_plot = df[df.location == radar_name]
        axs[i_radar].scatter(df_plot['ne_radar'], df_plot['ne_gf'])
        axs[i_radar].set_title('{radar}'.format(radar=radar_name), fontsize=14)
        axs[i_radar].set_xlim(-4e12, 5e12)
        axs[i_radar].set_ylim(-4e12, 5e12)
        axs[i_radar].axline((0, 1), slope=1, color='k', lw=0.75)

    mpl.rcParams.update({'font.size': 12})
    if mission == 'GR':
        plt.setp(axes[:, 0], ylabel="$Ne_{GR}$ [$m^{-3}$]")
    elif mission == 'GF':
        plt.setp(axes[:, 0], ylabel="$Ne_{GF}$ [$m^{-3}$]")

    plt.setp(axes[-1, :], xlabel="$Ne_{RADAR}$ [$m^{-3}$]")

    fig.tight_layout()
    # plt.show()
    plt.savefig("../figures/scatter_{mission}_individual.png".format(mission=mission))
    plt.close()

    # one big graph
    fig, ax = plt.subplots(figsize=(7, 7))

    for location in np.unique(df['location']):
        df_plot = df[df['location'] == location]
        plt.scatter(df_plot['ne_radar'], df_plot['ne_gf'], label=location)

    plt.xlim(-4e12, 5e12)
    plt.ylim(-4e12, 5e12)
    ax.axline((0, 1), slope=1, color='k', lw=0.75)
    plt.legend()

    plt.title('{mission_full_name} Conjunctions'.format(mission_full_name=mission_full_name), fontsize=16)
    plt.xlabel('$Ne_{RADAR}$ [$m^{-3}$]', fontsize=12)
    if mission == 'GR':
        plt.ylabel("$Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    elif mission == 'GF':
        plt.ylabel("$Ne_{GF}$ [$m^{-3}$]", fontsize=12)

    # plt.show()
    plt.savefig("../figures/scatter_{mission}_all.png".format(mission=mission))
    plt.close()

# plot_scatter('GR')
# plot_scatter('GF')


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

    df = pd.read_csv('../tables/{mission}_conjunctions_clean.csv'.format(mission=mission)).drop(['Unnamed: 0'], axis=1)

    df["radar_mnemonic_unique"] = df["radar_mnemonic"].str[0:3]

    df['location'] = df['radar_mnemonic_unique']
    df['longitude'] = df['radar_mnemonic_unique']

    for i in iter(range(0, len(df))):

        radar = df['radar_mnemonic_unique'][i]

        if str(radar)[0:3] == 'mlh':
            df['location'][i] = 'Millstone Hill'
            df['longitude'][i] = 288.51
        elif str(radar[0:3]) == 'arg':
            df['location'][i] = 'Arecibo'
            df['longitude'][i] = 293.25
        elif str(radar[0:3]) == 'arl':
            df['location'][i] = 'Arecibo'
            df['longitude'][i] = 293.25
        elif str(radar[0:3]) == 'eis':
            df['location'][i] = 'Tronsø'
            df['longitude'][i] = 19.21
        elif str(radar[0:3]) == 'tro':
            df['location'][i] = 'Tronsø'
            df['longitude'][i] = 19.21
        elif str(radar[0:3]) == 'jro':
            df['location'][i] = 'Jicamarca'
            df['longitude'][i] = 283.13
        elif str(radar[0:3]) == 'kpi':
            df['location'][i] = 'Kharkov'
            df['longitude'][i] = 36.2
        elif str(radar[0:3]) == 'mui':
            df['location'][i] = 'MU'
            df['longitude'][i] = 136.1
        elif str(radar[0:3]) == 'ran':
            df['location'][i] = 'Resolute Bay'
            df['longitude'][i] = 265.09424
        elif str(radar[0:3]) == 'ras':
            df['location'][i] = 'Resolute Bay'
            df['longitude'][i] = 265.09424
        elif str(radar[0:3]) == 'lyr':
            df['location'][i] = 'Svalbard'
            df['longitude'][i] = 16.02
        elif str(radar[0:3]) == 'pfa':
            df['location'][i] = 'Poker Flat'
            df['longitude'][i] = 212.529

    df = df[~df.location.str.contains("Kharkov", na=False)]
    df = df[~df.location.str.contains("MU", na=False)]

    df['date'] = pd.to_datetime(df['date'])

    df['longitude'] = df['longitude'].astype(float)

    df['date_local'] = df.apply(lambda x: calc_localtime(x.longitude, x.date), axis=1)

    df['hour'] = df.date.apply(lambda x: (x.hour + x.minute/60 + x.second/3600))

    # one big graph
    fig, ax = plt.subplots(figsize=(12, 5))

    for location in np.unique(df['location']):
        df_plot = df[df['location'] == location]
        plt.scatter(df_plot['hour'], df_plot['ne_diff'], label=location)

    plt.xlim(0,24)
    plt.xticks(np.arange(0,24))
    plt.legend()

    plt.title('{mission_full_name} Conjunctions'.format(mission_full_name=mission_full_name), fontsize=16)
    plt.xlabel('Local time [hours]', fontsize=12)
    if mission == 'GR':
        plt.ylabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)
    elif mission == 'GF':
        plt.ylabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)

    # plt.show()
    plt.savefig("../figures/timeday_{mission}_all.png".format(mission=mission))
    plt.close()

# plot_timeday('GR')
# plot_timeday('GF')


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

def plot_timeseries():

    gr = pd.read_csv('../tables/{mission}_conjunctions_clean.csv'.format(mission='GR')).drop(['Unnamed: 0'], axis=1)
    gf = pd.read_csv('../tables/{mission}_conjunctions_clean.csv'.format(mission='GF')).drop(['Unnamed: 0'], axis=1)

    df = pd.concat([gr,gf],ignore_index=True,sort=True)

    df["radar_mnemonic_unique"] = df["radar_mnemonic"].str[0:3]

    df['location'] = df['radar_mnemonic_unique']

    for i in iter(range(0, len(df))):

        radar = df['radar_mnemonic_unique'][i]

        if str(radar)[0:3] == 'mlh':
            df['location'][i] = 'Millstone Hill'
        elif str(radar[0:3]) == 'arg':
            df['location'][i] = 'Arecibo'
        elif str(radar[0:3]) == 'arl':
            df['location'][i] = 'Arecibo'
        elif str(radar[0:3]) == 'eis':
            df['location'][i] = 'Tronsø'
        elif str(radar[0:3]) == 'tro':
            df['location'][i] = 'Tronsø'
        elif str(radar[0:3]) == 'jro':
            df['location'][i] = 'Jicamarca'
        elif str(radar[0:3]) == 'kpi':
            df['location'][i] = 'Kharkov'
        elif str(radar[0:3]) == 'mui':
            df['location'][i] = 'Shigaraki'
        elif str(radar[0:3]) == 'ran':
            df['location'][i] = 'Resolute Bay'
        elif str(radar[0:3]) == 'ras':
            df['location'][i] = 'Resolute Bay'
        elif str(radar[0:3]) == 'lyr':
            df['location'][i] = 'Svalbard'
        elif str(radar[0:3]) == 'pfa':
            df['location'][i] = 'Poker Flat'

    df = df[~df.location.str.contains("Kharkov", na=False)]
    df = df[~df.location.str.contains("Shigaraki", na=False)]

    df['date'] = pd.to_datetime(df['date'])

    # one big graph
    fig, ax = plt.subplots(figsize=(18, 5))

    for location in np.unique(df['location']):
        df_plot = df[df['location'] == location]
        plt.scatter(df_plot['date'], df_plot['ne_diff'], label=location)

    ax.xaxis_date()

    months = mdates.MonthLocator(interval=12)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(pd.Timestamp("2002-01"), pd.Timestamp("2020-12"))

    plt.legend()

    plt.title('GRACE and GRACE-FO Conjunctions', fontsize=16)
    plt.xlabel('Dates', fontsize=12)
    plt.ylabel("Difference $Ne_{RADAR}$ - $Ne_{GR}$ [$m^{-3}$]", fontsize=12)

    # plt.show()
    plt.savefig("../figures/timeseries_all.png")
    plt.close()

# plot_timeseries()

