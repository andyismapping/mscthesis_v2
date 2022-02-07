import glob
import pandas as pd

gf_list = glob.glob('../data/interim/GRACEFO/KBRNE_relative_v2/dat/*')
gf_list.sort()

headerlist = ['CDF Epoch', 'GPS', 'Latitude', 'Longitude', 'Radius', 'Latitude_QD', 'Longitude_QD',
              'MLT', 'GRACE_1_Position_0', 'GRACE_1_Position_1', 'GRACE_1_Position_2', 'GRACE_2_Position_1',
              'GRACE_2_Position_2', 'GRACE_2_Position_3', 'Iono_Corr', 'Distance',
              'Relative_Hor_TEC', 'Relative_Ne']

for i in iter(range(0, len(gf_list))):

    if i == 0:
        gf = pd.read_csv(gf_list[i], sep='\s+', header=0, index_col=False, names=headerlist)
    else:
        gf_new = pd.read_csv(gf_list[i], sep='\s+', header=0, index_col=False, names=headerlist)
        gf = pd.concat([gf, gf_new])

gf = gf.drop_duplicates()
gf.to_csv('../data/processed/GF_ts_complete.csv')
