import pandas as pd

conjunction_df = pd.read_csv('../tables/v2/conjunctions.csv')

instMetadata = pd.read_csv('../data/external/Madrigal/instMetadata.csv')
radar_locations = pd.read_csv('../tables/clean_conjunctions_location.csv')

instMetadata_dict = instMetadata.set_index('Name').to_dict()['3-letter mnemonic']
instMetadata_dict.update({'Millstone Hill UHF Zenith Antenna, Millstone Hill UHF Steerable Antenna': 'mlh'})
instMetadata_dict.update({'EISCAT Troms   UHF IS radar': 'tro'})
instMetadata_dict.update({'Millstone Hill UHF Steerable Antenna, Millstone Hill UHF Zenith Antenna': 'mlh'})
instMetadata_dict.update({'Millstone Hill UHF Zenith Antenna,Millstone Hill UHF Steerable Antenna': 'mlh'})

conjunction_df['radar_mnemonic'] = conjunction_df['radar'].map(instMetadata_dict)

conjunction_df_mean = conjunction_df.groupby(['date']).mean()

conjunction_df_join = conjunction_df.groupby(['date'], as_index=False, sort=False).agg(' , '.join)

conjunction_df_clean = conjunction_df_mean.merge(conjunction_df_join, left_on='date', right_on='date')

conjunction_df_clean["radar_mnemonic_unique"] = conjunction_df_clean["radar_mnemonic"].str[0:3]
conjunction_df_clean["mission_unique"] = conjunction_df_clean["mission"].str[0:2]

conjunction_df_clean = conjunction_df_clean[~conjunction_df_clean.radar_mnemonic_unique.str.contains("kpi", na=False)]
conjunction_df_clean = conjunction_df_clean[~conjunction_df_clean.radar_mnemonic_unique.str.contains("mui", na=False)]

latitude_dict = radar_locations.set_index('radar_mnemonic').to_dict()['Latitude']
longitude_dict = radar_locations.set_index('radar_mnemonic').to_dict()['Longitude']
location_dict = radar_locations.set_index('radar_mnemonic').to_dict()['Name']

conjunction_df_clean['Latitude'] = conjunction_df_clean['radar_mnemonic_unique'].map(latitude_dict)
conjunction_df_clean['Longitude'] = conjunction_df_clean['radar_mnemonic_unique'].map(longitude_dict)
conjunction_df_clean['Location'] = conjunction_df_clean['radar_mnemonic_unique'].map(location_dict)

conjunction_df_clean.to_csv('../tables/v2/conjunctions_clean.csv')


GR = conjunction_df_clean[conjunction_df_clean['mission_unique'] == 'GR']
GR.to_csv('../tables/v2/conjunctions_clean_GR.csv')
# GR.Location.value_counts().sort_index()
# GR.groupby('Location').mean()['nel_diff']
# GR.groupby('Location').std()['nel_diff']

GF = conjunction_df_clean[conjunction_df_clean['mission_unique'] == 'GF']
GF.to_csv('../tables/v2/conjunctions_clean_GF.csv')
# GF.Location.value_counts().sort_index()
# GF.groupby('Location').mean()['nel_diff']
# GF.groupby('Location').std()['nel_diff']
