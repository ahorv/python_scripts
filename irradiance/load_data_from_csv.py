import pandas as pd


######################################################################
## Hoa: 08.11.2018 Version 1 : load_data_from_csv.py
######################################################################
# process_LUZ()
# Reads a *.csv file containing irradiation data from a weatherstation
# near Lucern 47°02'11.0"N 8°18'04.0"E at 454 m.
#
# Data contained:
#  - stn: Stations id
#  - time: date
#  - gor000za: Global irradiation standart deviation in W/m²
#  - gre000z0: mean global irradiation of 10 minutes
#####################################################################
# process_SODA()
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 08.11.2018 : first add
#
######################################################################

path_luz = r'irradiation_luz_2017_2018.csv'
path_soda = r'irradiation_soda_2017_2018_1min.csv'
#path_soda = r'irradiation_soda_2017_2018.csv'


def process_LUZ(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', utc=True)
    df['time'] = df['time'].dt.tz_localize('UTC')
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['gre000z0'] = pd.to_numeric(df['gre000z0'], errors='coerce')

    #df.to_csv('luzout.csv')
    return df

def process_SODA(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=36)
    df.rename(columns={'Observation period': 'datetime'}, inplace=True)
    df['datetime'] = df['datetime'].map(lambda x: (((x.split('/')[1]).replace('T',' ')).replace('.0','')))
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)
    df.set_index(pd.DatetimeIndex(df['datetime']).tz_localize('UTC').tz_convert('Europe/Zurich'), inplace=True)
    df = df.loc[df.index.minute % 10 == 0]
    del df['datetime']

    #df.to_csv('sodaout_10min.csv')
    return df


'''
def main():
    try:
        df = process_LUZ(path_luz)
        #df = process_SODA(path_soda)   
        print(df.head(n=10))


    except Exception as e:
       print('Error in MAIN: {}'.format(e))

if __name__ == '__main__':
    main()
'''
