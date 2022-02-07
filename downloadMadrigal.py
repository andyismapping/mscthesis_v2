import os

df_inst = pd.read_csv('../data/external/Madrigal/madrigal/instMetadata.csv')

list_inst= [21,20,22,70,71,75,73,76,95,72,74,10,45,30,25,61,92,91,80,100]

os.system("conda activate MScThesis")

for inst in list_inst:
    # find inst id and get the name for the folder

    name = name.replace(" ", "")
    os.system('globalDownload.py --verbose --url=http://cedar.openmadrigal.org --outputDir=/Users/andyaracallegare/Documents/madrigal_redownload/{name} "--user_fullname="Andyara Callegare” --user_email=oliveiracallegare@uni-potsdam.de --user_affiliation="UP" --format="hdf5" --startDate="01/01/2000" --endDate="12/31/2021" --inst={kinst}'.format(name=name,kinst = inst)